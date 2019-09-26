from __future__ import print_function

import heapq
import random
import time

from collections import namedtuple

import numpy as np

from examples.pybullet.utils.pybullet_tools.utils import elapsed_time, \
    remove_all_debug, wait_for_user, has_gui, LockRenderer, reset_simulation, disconnect, set_renderer
from extrusion.parsing import load_extrusion, draw_element, draw_sequence, draw_model
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, get_id_from_element, load_world, get_supported_orders, get_extructed_ids

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import connect, ClientSaver, wait_for_user, INF, get_distance, has_gui, remove_all_debug
from pddlstream.utils import neighbors_from_orders, adjacent_from_edges, implies

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

GREEDY_HEURISTICS = [
    None,
    'z',
    #'dijkstra',
    #'fixed-dijkstra',
    'stiffness', # Performs poorly with respect to stiffness
    'fixed-stiffness',
    'relative-stiffness',
    'length',
    'degree',
]

# TODO: visualize the branching factor

def get_heuristic_fn(extrusion_path, heuristic, forward, checker=None):
    # TODO: penalize disconnected
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    elements = set(element_from_id.values())
    distance_from_node = compute_distance_from_node(elements, node_points, ground_nodes)
    sign = +1 if forward else -1

    stiffness_cache = {}
    if heuristic in ('fixed-stiffness', 'relative-stiffness'):
        stiffness_cache.update({element: score_stiffness(extrusion_path, element_from_id, elements - {element},
                                                         checker=checker) for element in elements})

    def fn(printed, element):
        # Queue minimizes the statistic
        if heuristic is None:
            return 0
        elif heuristic == 'degree':
            # TODO: online/offline and ground
            #printed_nodes = {n for e in printed for n in e}
            #n1, n2 = element
            #node = n1 if n2 in printed_nodes else n2
            #if node in ground_nodes:
            #    return 0
            raise NotImplementedError()
        elif heuristic == 'length':
            # Equivalent to mass if uniform density
            n1, n2 = element
            return get_distance(node_points[n2], node_points[n1])
        elif heuristic == 'z':
            # TODO: round values for more tie-breaking opportunities
            z = get_z(node_points, element)
            return sign*z
        elif heuristic == 'stiffness':
            # TODO: add different variations
            # TODO: normalize by initial stiffness, length, or degree
            # Most unstable or least unstable first
            # Gets faster with fewer elements
            #old_stiffness = score_stiffness(extrusion_path, element_from_id, printed, checker=checker)
            structure = printed | {element} if forward else printed - {element}
            stiffness = score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
            return stiffness
            #return stiffness / old_stiffness
        elif heuristic == 'fixed-stiffness':
            # TODO: invert the sign for regression/progression?
            # TODO: sort FastDownward by the (fixed) action cost
            return stiffness_cache[element]
        elif heuristic == 'relative-stiffness':
            structure = printed | {element} if forward else printed - {element}
            stiffness = score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
            length = sum(get_distance(node_points[n2], node_points[n1]) for n1, n2 in structure)
            if length == 0:
                return 0
            return stiffness / length
            #return stiffness / stiffness_cache[element]
        elif heuristic == 'dijkstra':
            # min, max, node not in set
            # TODO: recompute online (but all at once)
            # TODO: sum of all element path distances
            distance = np.average([distance_from_node[node] for node in element])
            return sign * distance
        raise ValueError(heuristic)
    return fn

def get_z(node_points, element):
    # TODO: tiebreak by angle or x
    return np.average([node_points[n][2] for n in element])

def compute_distance_from_node(elements, node_points, ground_nodes):
    #incoming_supporters, _ = neighbors_from_orders(get_supported_orders(
    #    element_from_id.values(), node_points))
    neighbors = adjacent_from_edges(elements)
    edge_costs = {edge: get_distance(node_points[edge[0]], node_points[edge[1]])
                  for edge in elements}
    edge_costs.update({edge[::-1]: distance for edge, distance in edge_costs.items()})

    cost_from_node = {}
    queue = []
    for node in ground_nodes:
        cost = 0
        cost_from_node[node] = cost
        heapq.heappush(queue, (cost, node))
    while queue:
        cost1, node1 = heapq.heappop(queue)
        if cost_from_node[node1] < cost1:
            continue
        for node2 in neighbors[node1]:
            cost2 = cost1 + edge_costs[node1, node2]
            if cost2 < cost_from_node.get(node2, INF):
                cost_from_node[node2] = cost2
                heapq.heappush(queue, (cost2, node2))
    return cost_from_node

def score_stiffness(extrusion_path, element_from_id, elements, checker=None):
    if not elements:
        return 0
    if checker is None:
        checker = create_stiffness_checker(extrusion_path)
    # TODO: analyze fixities projections in the xy plane

    # Lower is better
    extruded_ids = get_extructed_ids(element_from_id, elements)
    checker.solve(exist_element_ids=extruded_ids, if_cond_num=True)
    success, nodal_displacement, fixities_reaction, _ = checker.get_solved_results()
    if not success:
        return INF
    #operation = np.max
    operation = np.sum # equivalently average
    # TODO: LInf or L1 norm applied on forces
    # TODO: looking for a colored path through the space

    # trans unit: meter, rot unit: rad
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    max_trans, max_rot, _, _ = checker.get_max_nodal_deformation()
    relative_trans = max_trans / trans_tol # lower is better
    relative_rot = max_rot / rot_tol # lower is better
    # More quickly approximate deformations by modifying the matrix operations incrementally

    reaction_forces = np.array([d[:3] for d in fixities_reaction.values()])
    reaction_moments = np.array([d[3:] for d in fixities_reaction.values()])
    heuristic = 'fixities_rotation'
    scores = {
        # Yijiang was suprised that fixities_translation worked
        'fixities_translation': np.linalg.norm(reaction_forces, axis=1),
        'fixities_rotation': np.linalg.norm(reaction_moments, axis=1),
        'nodal_translation': np.linalg.norm(list(nodal_displacement.values()), axis=1),
        'compliance': [checker.get_compliance()],
        'deformation': [relative_trans, relative_rot],
    }
    # TODO: remove pairs of elements
    # TODO: clustering
    return operation(scores[heuristic])
    #return relative_trans
    #return max(relative_trans, relative_rot)
    #return relative_trans + relative_rot # arithmetic mean
    #return relative_trans * relative_rot # geometric mean
    #return 2*relative_trans * relative_rot / (relative_trans + relative_rot) # harmonic mean

##################################################

def progression(robot, obstacles, element_bodies, extrusion_path,
                heuristic='z', max_time=INF, max_backtrack=INF, stiffness=True, **kwargs):

    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)

    queue = []
    visited = {}
    def add_successors(printed):
        remaining = elements - printed
        for element in sorted(remaining, key=lambda e: get_z(node_points, e)):
            num_remaining = len(remaining) - 1
            assert 0 <= num_remaining
            bias = heuristic_fn(printed, element)
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, printed, element))

    final_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, final_printed) or \
            not test_stiffness(extrusion_path, element_from_id, final_printed):
        data = {
            'sequence': None,
            'runtime': elapsed_time(start_time),
        }
        return None, data

    initial_printed = frozenset()
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    if has_gui():
        sequence = sorted(initial_printed, key=lambda e: heuristic_fn(initial_printed, e), reverse=True)
        remove_all_debug()
        draw_sequence(sequence, node_points)
        wait_for_user()

    plan = None
    min_remaining = INF
    num_evaluated = 0
    while queue and (elapsed_time(start_time) < max_time):
        num_evaluated += 1
        _, printed, element = heapq.heappop(queue)
        num_remaining = len(elements) - len(printed)
        backtrack = num_remaining - min_remaining
        if max_backtrack <= backtrack:
            continue
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
        if elements <= next_printed:
            min_remaining = 0
            plan = retrace_plan(visited, next_printed)
            break
        add_successors(next_printed)

    sequence = None
    if plan is not None:
        sequence = [traj.element for traj in plan]
    data = {
        'sequence': sequence,
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_remaining,
        'num_elements': len(elements)
    }
    return plan, data

##################################################

def regression(robot, obstacles, element_bodies, extrusion_path,
               heuristic='z', max_time=INF, max_backtrack=INF, stiffness=True, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: persistent search to reuse
    # TODO: max branching factor

    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    id_from_element = get_id_from_element(element_from_id)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=False)
    # TODO: compute the heuristic function once and fix

    queue = []
    visited = {}
    def add_successors(printed):
        for element in sorted(printed, key=lambda e: -get_z(node_points, e)):
            num_remaining = len(printed) - 1
            assert 0 <= num_remaining
            bias = heuristic_fn(printed, element)
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, printed, element))

    initial_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, initial_printed) or \
            not test_stiffness(extrusion_path, element_from_id, initial_printed, checker=checker):
        data = {
            'sequence': None,
            'runtime': elapsed_time(start_time),
        }
        return None, data
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    # TODO: lazy hill climbing using the true stiffness heuristic
    # Choose the first move that improves the score
    if has_gui():
        sequence = sorted(initial_printed, key=lambda e: heuristic_fn(initial_printed, e), reverse=True)
        remove_all_debug()
        draw_sequence(sequence, node_points)
        wait_for_user()
    # TODO: fixed branching factor
    # TODO: be more careful when near the end
    # TODO: max time spent evaluating successors (less expensive when few left)
    # TODO: tree rollouts
    # TODO: best-first search with a minimizing path distance cost
    # TODO: immediately select if becomes more stable
    # TODO: focus branching factor on most stable regions

    plan = None
    min_remaining = INF
    num_evaluated = 0
    while queue and (elapsed_time(start_time) < max_time):
        priority, printed, element = heapq.heappop(queue)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        if max_backtrack <= backtrack:
            continue
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining
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
        command = sample_extrusion(print_gen_fn, ground_nodes, next_printed, element)
        if command is None:
            continue
        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            plan = list(reversed(retrace_plan(visited, next_printed)))
            break
        add_successors(next_printed)

    # TODO: store maximum stiffness violations (for speed purposes)
    sequence = None
    if plan is not None:
        sequence = [traj.element for traj in plan]
    data = {
        'sequence': sequence,
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_remaining,
        'num_elements': len(element_bodies)
    }
    return plan, data

GREEDY_ALGORITHMS = [
    progression.__name__,
    regression.__name__,
]