from __future__ import print_function

import heapq
import random
import time

from collections import namedtuple

import numpy as np

from examples.pybullet.utils.pybullet_tools.utils import elapsed_time, \
    remove_all_debug, wait_for_user, has_gui, LockRenderer
from extrusion.parsing import load_extrusion, draw_element
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, check_stiffness, \
    create_stiffness_checker, score_stiffness, get_id_from_element, load_world

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import connect, ClientSaver, wait_for_user, INF

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

def get_z(node_points, element):
    # TODO: tiebreak by angle or x
    return np.average([node_points[n][2] for n in element])


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

def regression(robot, obstacles, element_bodies, extrusion_name,
               max_time=INF, max_backtrack=INF, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: persistent search to reuse
    # TODO: max branching factor

    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_name)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
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
        draw_action(node_points, next_printed, element)
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not check_stiffness(extrusion_name, element_from_id, next_printed):
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
    # TODO: different heuristics
    # TODO: investigate recovering structure support from conmech

    data = {
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_printed,
        'num_elements': len(element_bodies)
    }
    return plan, data

##################################################

def progression(robot, obstacles, element_bodies, extrusion_name,
                max_time=INF, max_backtrack=INF, **kwargs):

    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_name)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=500, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
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
                not check_stiffness(extrusion_name, element_from_id, next_printed):
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
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_printed,
        'num_elements': len(elements)
    }
    return plan, data
