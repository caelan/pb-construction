import heapq
import random
import time
from collections import namedtuple

from examples.pybullet.utils.pybullet_tools.utils import elapsed_time, \
    remove_all_debug, wait_for_user, has_gui, LockRenderer
from extrusion.parsing import load_extrusion, draw_element
from extrusion.sorted import get_z
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, check_stiffness, \
    create_stiffness_checker, score_stiffness, get_id_from_element

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp

State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])

def retrace_plan(visited, current_state):
    command, prev_state = visited[current_state]
    if prev_state is None:
        return []
    return retrace_plan(visited, prev_state) + [traj for traj in command.trajectories]

    # TODO: search over local stability for each node
    # TODO: progression search

##################################################

def sample_extrusion(print_gen_fn, ground_nodes, printed, element):
    next_nodes = {n for e in printed for n in e} | set(ground_nodes)
    for node in element:
        if node in next_nodes:
            try:
                command, = next(print_gen_fn(node, element, extruded=printed))
                return command
            except StopIteration:
                pass
    return None

##################################################

def draw_action(node_points, printed, element):
    if not has_gui():
        return []
    with LockRenderer():
        remove_all_debug()
        handles = [draw_element(node_points, element, color=(1, 0, 0))]
        handles.extend(draw_element(node_points, e, color=(0, 1, 0)) for e in printed)
    wait_for_user()
    return handles

def regression(robot, obstacles, element_bodies, extrusion_name, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: persistent search to reuse
    # TODO: max branching factor
    # TODO: persistent search

    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_name)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    #checker = create_stiffness_checker(extrusion_name)
    id_from_element = get_id_from_element(element_from_id)

    queue = []
    visited = {}
    def add_successors(printed):
        for element in sorted(printed, key=lambda e: -get_z(node_points, e)):
            num_remaining = len(printed) - 1
            assert 0 <= num_remaining
            bias = -get_z(node_points, element)
            #bias = 0
            #bias = score_stiffness(extrusion_name, element_from_id, printed - {element})
            # TODO: penalize disconnected
            #print(bias)
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, printed, element))

    initial_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, initial_printed) or \
            not check_stiffness(extrusion_name, element_from_id, initial_printed):
        return None
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    #N = 1000
    #start_time = time.time() # 0.00717480421066
    #for _ in range(N):
    #    check_stiffness(checker, element_from_id, initial_printed)
    #print(elapsed_time(start_time) / N)

    start_time = time.time()
    iteration = 0
    while queue:
        iteration += 1
        priority, printed, element = heapq.heappop(queue)
        print(priority)
        print('Iteration: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            iteration, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed - {element}
        draw_action(node_points, next_printed, element)
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not check_stiffness(extrusion_name, element_from_id, next_printed):
            continue
        command = sample_extrusion(print_gen_fn, ground_nodes, next_printed, element)
        if command is None:
            continue
        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            return list(reversed(retrace_plan(visited, next_printed)))
        add_successors(next_printed)
    return None

##################################################

def progression(robot, obstacles, element_bodies, extrusion_name, **kwargs):
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_name)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    checker = create_stiffness_checker(extrusion_name)
    elements = frozenset(element_bodies)

    queue = []
    visited = {}
    def add_successors(printed):
        remaining = elements - printed
        for element in sorted(remaining, key=lambda e: get_z(node_points, e)):
            num_remaining = len(remaining) - 1
            assert 0 <= num_remaining
            priority = (num_remaining, get_z(node_points, element), random.random())
            heapq.heappush(queue, (priority, printed, element))

    initial_printed = frozenset()
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    start_time = time.time()
    iteration = 0
    while queue:
        iteration += 1
        _, printed, element = heapq.heappop(queue)
        print('Iteration: {} | Printed: {} | Element: {} | Time: {:.3f}'.format(
            iteration, len(printed), element, elapsed_time(start_time)))
        next_printed = printed | {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not check_stiffness(checker, element_from_id, next_printed):
            continue
        command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        if command is None:
            continue
        visited[next_printed] = Node(command, printed)
        if elements <= next_printed:
            return retrace_plan(visited, next_printed)
        add_successors(next_printed)
    return None