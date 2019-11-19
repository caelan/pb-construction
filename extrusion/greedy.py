from __future__ import print_function

import heapq
import random
import time

from collections import namedtuple

from extrusion.validator import compute_plan_deformation
from extrusion.heuristics import get_heuristic_fn
from pybullet_tools.utils import elapsed_time, \
    LockRenderer, reset_simulation, disconnect, randomize
from extrusion.parsing import load_extrusion
from extrusion.visualization import draw_element, draw_ordered
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, get_id_from_element, load_world, nodes_from_elements, PrintTrajectory

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import connect, ClientSaver, wait_for_user, INF, has_gui, remove_all_debug
from pddlstream.utils import implies

#State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])

##################################################

def retrace_trajectories(visited, current_state, reverse=False):
    command, prev_state = visited[current_state]
    if prev_state is None:
        return []
    prior_trajectories = retrace_trajectories(visited, prev_state, reverse)
    current_trajectories = [traj for traj in command.trajectories]
    if reverse:
        return current_trajectories + prior_trajectories
    return prior_trajectories + current_trajectories
    # TODO: search over local stability for each node

def recover_sequence(plan):
    if plan is None:
        return plan
    return [traj.element for traj in plan if isinstance(traj, PrintTrajectory)]

def recover_directed_sequence(plan):
    if plan is None:
        return plan
    return [traj.directed_element for traj in plan if isinstance(traj, PrintTrajectory)]

##################################################

def compute_printed_nodes(ground_nodes, printed):
    return nodes_from_elements(printed) | set(ground_nodes)

def compute_printable_elements(elements, ground_nodes, printed):
    nodes = compute_printed_nodes(ground_nodes, printed)
    return {element for element in set(elements) - set(printed)
            if any(n in nodes for n in element)}

def sample_extrusion(print_gen_fn, ground_nodes, printed, element):
    printed_nodes = compute_printed_nodes(ground_nodes, printed)
    # TODO: could always reverse these trajectories
    for node in element:
        # TODO: sample between different orientations if both are feasible
        if node in printed_nodes:
            try:
                command, = next(print_gen_fn(node, element, extruded=printed))
                return command
            except StopIteration:
                pass
    return None

def display_failure(node_points, extruded_elements, element):
    client = connect(use_gui=True)
    with ClientSaver(client):
        obstacles, robot = load_world()
        handles = []
        for e in extruded_elements:
            handles.append(draw_element(node_points, e, color=(0, 1, 0)))
        handles.append(draw_element(node_points, element, color=(1, 0, 0)))
        print('Failure!')
        wait_for_user()
        reset_simulation()
        disconnect()

def draw_action(node_points, printed, element):
    if not has_gui():
        return []
    with LockRenderer():
        remove_all_debug()
        handles = [draw_element(node_points, element, color=(1, 0, 0))]
        handles.extend(draw_element(node_points, e, color=(0, 1, 0)) for e in printed)
    wait_for_user()
    return handles

##################################################

def add_successors(queue, elements, node_points, ground_nodes, heuristic_fn, printed, conf,
                   visualize=False):
    remaining = elements - printed
    num_remaining = len(remaining) - 1
    assert 0 <= num_remaining
    bias_from_element = {}
    for element in randomize(compute_printable_elements(elements, ground_nodes, printed)):
        bias = heuristic_fn(printed, element, conf)
        priority = (num_remaining, bias, random.random())
        visits = 0
        heapq.heappush(queue, (visits, priority, printed, element, conf))
        bias_from_element[element] = bias

    if visualize and has_gui():
        handles = []
        with LockRenderer():
            remove_all_debug()
            for element in printed:
                handles.append(draw_element(node_points, element, color=(0, 0, 0)))
            successors = sorted(bias_from_element, key=lambda e: bias_from_element[e])
            handles.extend(draw_ordered(successors, node_points))
        print('Min: {:.3E} | Max: {:.3E}'.format(bias_from_element[successors[0]], bias_from_element[successors[-1]]))
        wait_for_user()

def progression(robot, obstacles, element_bodies, extrusion_path,
                heuristic='z', max_time=INF, # max_backtrack=INF,
                stiffness=True, **kwargs):

    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, bidirectional=False,
                                    precompute_collisions=False, max_directions=500, max_attempts=1, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)

    initial_conf = None
    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    if check_connected(ground_nodes, all_elements) and \
            test_stiffness(extrusion_path, element_from_id, all_elements):
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, initial_printed, initial_conf)

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = 0
    while queue and (elapsed_time(start_time) < max_time):
        num_evaluated += 1
        visits, _, printed, element, current_conf = heapq.heappop(queue)
        num_remaining = len(all_elements) - len(printed)
        backtrack = num_remaining - min_remaining
        max_backtrack = max(max_backtrack, backtrack)
        #if max_backtrack <= backtrack:
        #    continue
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining
        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed | {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                (stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            continue
        command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        if command is None:
            continue
        visited[next_printed] = Node(command, printed)
        if all_elements <= next_printed:
            min_remaining = 0
            plan = retrace_trajectories(visited, next_printed)
            break
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, next_printed, initial_conf)

    max_translation, max_rotation = compute_plan_deformation(extrusion_path, recover_sequence(plan))
    data = {
        'sequence': recover_directed_sequence(plan),
        'runtime': elapsed_time(start_time),
        'num_elements': len(all_elements),
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        'max_translation': max_translation,
        'max_rotation': max_rotation,
    }
    return plan, data

##################################################

def regression(robot, obstacles, element_bodies, extrusion_path,
               heuristic='z', max_time=INF, # max_backtrack=INF,
               stiffness=True, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: persistent search
    # TODO: max branching factor

    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, bidirectional=False,
                                    precompute_collisions=False, max_directions=500, max_attempts=1, **kwargs)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=False)

    final_conf = None
    initial_printed = all_elements
    queue = []
    visited = {initial_printed: Node(None, None)}

    def add_successors(printed):
        for element in randomize(printed):
            num_remaining = len(printed) - 1
            assert 0 <= num_remaining
            bias = heuristic_fn(printed, element, conf=None)
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, printed, element))

    if check_connected(ground_nodes, initial_printed) and \
            test_stiffness(extrusion_path, element_from_id, initial_printed, checker=checker):
        add_successors(initial_printed)

    # TODO: lazy hill climbing using the true stiffness heuristic
    # Choose the first move that improves the score
    if has_gui():
        sequence = sorted(initial_printed, key=lambda e: heuristic_fn(initial_printed, e, conf=None), reverse=True)
        remove_all_debug()
        draw_ordered(sequence, node_points)
        wait_for_user()
    # TODO: fixed branching factor
    # TODO: be more careful when near the end
    # TODO: max time spent evaluating successors (less expensive when few left)
    # TODO: tree rollouts
    # TODO: best-first search with a minimizing path distance cost
    # TODO: immediately select if becomes more stable
    # TODO: focus branching factor on most stable regions

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = 0
    while queue and (elapsed_time(start_time) < max_time):
        priority, printed, element = heapq.heappop(queue)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        max_backtrack = max(max_backtrack, backtrack)
        #if max_backtrack <= backtrack:
        #    continue
        num_evaluated += 1

        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed - {element}
        #draw_action(node_points, next_printed, element)
        #if 3 < backtrack + 1:
        #    remove_all_debug()
        #    set_renderer(enable=True)
        #    draw_model(next_printed, node_points, ground_nodes)
        #    wait_for_user()

        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not implies(stiffness, test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            continue
        # TODO: could do this eagerly to inspect the full branching factor
        command = sample_extrusion(print_gen_fn, ground_nodes, next_printed, element)
        if command is None:
            continue

        if num_remaining < min_remaining:
            min_remaining = num_remaining
            print('New best: {}'.format(num_remaining))
            #if has_gui():
            #    # TODO: change link transparency
            #    remove_all_debug()
            #    draw_model(next_printed, node_points, ground_nodes)
            #    wait_for_duration(0.5)

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            plan = retrace_trajectories(visited, next_printed, reverse=True)
            break
        add_successors(next_printed)

    max_translation, max_rotation = compute_plan_deformation(extrusion_path, recover_sequence(plan))
    data = {
        'sequence': recover_directed_sequence(plan),
        'runtime': elapsed_time(start_time),
        'num_elements': len(all_elements),
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        'max_translation': max_translation,
        'max_rotation': max_rotation,
    }
    return plan, data

##################################################

GREEDY_ALGORITHMS = [
    progression.__name__,
    regression.__name__,
]
