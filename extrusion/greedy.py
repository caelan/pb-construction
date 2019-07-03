from __future__ import print_function

import heapq
import random
import time

from collections import namedtuple

import numpy as np

from examples.pybullet.utils.pybullet_tools.utils import elapsed_time, \
    remove_all_debug, wait_for_user, has_gui, LockRenderer, reset_simulation, disconnect
from extrusion.parsing import load_extrusion, draw_element
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, score_stiffness, get_id_from_element, load_world, get_supported_orders

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import connect, ClientSaver, wait_for_user, INF, get_distance
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
        floor, robot = load_world()
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

HEURISTICS = [None, 'z', 'dijkstra', 'stiffness']

def get_heuristic_fn(extrusion_path, heuristic, forward):
    # TODO: penalize disconnected
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    distance_from_node = compute_distance_from_node(element_from_id.values(), node_points, ground_nodes)
    sign = +1 if forward else -1
    def fn(printed, element):
        if heuristic is None:
            return 0
        elif heuristic == 'z':
            z = get_z(node_points, element)
            return sign*z
        elif heuristic == 'stiffness':
            structure = printed | {element} if forward else printed - {element}
            return score_stiffness(extrusion_path, element_from_id, structure)
        elif heuristic == 'dijkstra':
            # min, max, node not in set
            distance = np.average([distance_from_node[node] for node in element])
            return sign * distance
            # Could also recompute online
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

##################################################

def progression(robot, obstacles, element_bodies, extrusion_path,
                heuristic='z', max_time=INF, max_backtrack=INF, stiffness=True, **kwargs):

    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, forward=True)

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

    initial_printed = frozenset()
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    plan = None
    min_printed = INF
    start_time = time.time()
    num_evaluated = 0
    while queue and (elapsed_time(start_time) < max_time):
        num_evaluated += 1
        _, printed, element = heapq.heappop(queue)
        num_remaining = len(elements) - len(printed)
        backtrack = num_remaining - min_printed
        if max_backtrack <= backtrack:
            continue
        num_evaluated += 1
        if len(printed) < min_printed:
            min_printed = len(printed)
        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_printed, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed | {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                (stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed)):
            continue
        command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        if command is None:
            continue
        visited[next_printed] = Node(command, printed)
        if elements <= next_printed:
            min_printed = 0
            plan = retrace_plan(visited, next_printed)
            break
        add_successors(next_printed)

    data = {
        'success': plan is not None,
        'length': INF if plan is None else len(plan),
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_printed,
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

    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, forward=False)

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
            not test_stiffness(extrusion_path, element_from_id, initial_printed):
        return None
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    plan = None
    min_printed = INF
    start_time = time.time()
    num_evaluated = 0
    while queue and (elapsed_time(start_time) < max_time):
        priority, printed, element = heapq.heappop(queue)
        num_remaining = len(printed)
        backtrack = num_remaining - min_printed
        if max_backtrack <= backtrack:
            continue
        num_evaluated += 1
        if len(printed) < min_printed:
            min_printed = len(printed)
        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_printed, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed - {element}
        #draw_action(node_points, next_printed, element)
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not implies(stiffness, test_stiffness(extrusion_path, element_from_id, next_printed)):
            continue
        command = sample_extrusion(print_gen_fn, ground_nodes, next_printed, element)
        if command is None:
            continue
        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_printed = 0
            plan = list(reversed(retrace_plan(visited, next_printed)))
            break
        add_successors(next_printed)

    # TODO: parallelize

    data = {
        'success': plan is not None,
        'length': INF if plan is None else len(plan),
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_printed,
        'num_elements': len(element_bodies)
    }
    return plan, data
