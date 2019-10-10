from __future__ import print_function

import heapq
import time
from collections import defaultdict

from extrusion.greedy import get_heuristic_fn, Node, retrace_plan, add_successors, compute_printed_nodes
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, get_id_from_element
from extrusion.visualization import color_structure
# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import INF, has_gui
from pybullet_tools.utils import elapsed_time, \
    LockRenderer, randomize


def get_sample_traj(elements, print_gen_fn):
    gen_from_element = {element: print_gen_fn(None, element) for element in elements}
    trajs_from_element = defaultdict(list)

    def enumerate_extrusions(element):
        for traj in trajs_from_element[element]:
            yield traj
        with LockRenderer():
            for traj, in gen_from_element[element]: # TODO: islice for the num to sample
                trajs_from_element[element].append(traj)
                yield traj

    def sample_traj(printed, element):
        # TODO: condition on the printed elements
        for traj in enumerate_extrusions(element):
            # TODO: lazy collision checking
            if not (traj.colliding & printed):
                return traj
        return None
    return sample_traj

def deadend(robot, obstacles, element_bodies, extrusion_path,
            heuristic='z', max_time=INF, max_backtrack=INF, stiffness=True, **kwargs):
    # TODO: persistent search that revisits prunning heuristic
    # TODO: check for feasible end-effector trajectories
    # TODO: prioritize edges that remove few trajectories

    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, supports=False, bidirectional=False, ee_only=False,
                                    max_directions=50, max_attempts=10, **kwargs)
    full_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                         precompute_collisions=True, supports=False, bidirectional=True, ee_only=False,
                                         max_directions=50, max_attempts=10, **kwargs)
    # TODO: could just check kinematics instead of collision
    ee_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                        precompute_collisions=True, supports=False, bidirectional=True, ee_only=True,
                                        max_directions=50, max_attempts=10, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)

    final_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, final_printed) or \
            not test_stiffness(extrusion_path, element_from_id, final_printed):
        data = {
            'sequence': None,
            'runtime': elapsed_time(start_time),
        }
        return None, data

    full_sample_traj = get_sample_traj(elements, full_print_gen_fn)
    ee_sample_traj = get_sample_traj(elements, ee_print_gen_fn)

    def sample_remaining(printed, sample_fn):
        # TODO: only check nodes that can be immediately printed?
        return all(sample_fn(printed, element) is not None for element in randomize(elements - printed))

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    add_successors(queue, elements, node_points, ground_nodes, heuristic_fn, initial_printed)

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
        printed_nodes = compute_printed_nodes(ground_nodes, printed)
        if has_gui():
            color_structure(element_bodies, printed, element)

        next_printed = printed | {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                (stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            # Hard dead-end
            continue

        # TODO: before or after sampling?
        # Constraint propagation
        # forward checking / lookahead: prove infeasibility quickly
        # https://en.wikipedia.org/wiki/Look-ahead_(backtracking)
        if not sample_remaining(next_printed, ee_sample_traj):
            # Soft dead-end
            continue

        #command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        command = full_sample_traj(printed, element)
        if command is None:
            # Soft dead-end
            continue
        if command.trajectories[0].n1 not in printed_nodes:
            command = command.reverse()
        # TODO: sample several EE trajectories and then sort by non-dominated

        if not sample_remaining(next_printed, full_sample_traj):
            # Soft dead-end
            continue

        # TODO: test end-effector here first
        # TODO: separate parameters for dead-end versus transition
        visited[next_printed] = Node(command, printed)
        if elements <= next_printed:
            min_remaining = 0
            plan = retrace_plan(visited, next_printed)
            break
        add_successors(queue, elements, node_points, ground_nodes, heuristic_fn, next_printed)

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