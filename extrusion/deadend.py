from __future__ import print_function

import heapq
import time
import numpy as np
from collections import defaultdict

from extrusion.greedy import get_heuristic_fn, Node, retrace_plan, add_successors, compute_printed_nodes
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, get_id_from_element
from extrusion.visualization import color_structure
# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import INF, has_gui, elapsed_time, LockRenderer, randomize, wait_for_user


def get_sample_traj(elements, print_gen_fn):
    gen_from_element = {element: print_gen_fn(None, element, extruded=[], trajectories=[]) for element in elements}
    trajs_from_element = defaultdict(list)

    def enumerate_extrusions(element):
        for traj in trajs_from_element[element]:
            yield traj
        with LockRenderer():
            for traj, in gen_from_element[element]: # TODO: islice for the num to sample
                trajs_from_element[element].append(traj)
                yield traj
            #for _ in range(100):
            #    traj, = next(print_gen_fn(None, element, extruded=[]), (None,))


    def sample_traj(printed, element):
        # TODO: condition on the printed elements
        for traj in enumerate_extrusions(element):
            # TODO: lazy collision checking
            if not (traj.colliding & printed):
                return traj
        return None
    return sample_traj, trajs_from_element

def lookahead(robot, obstacles, element_bodies, extrusion_path,
              heuristic='z', max_time=INF, max_backtrack=INF,
              ee_only=False, stiffness=True, steps=1, **kwargs):
    # TODO: persistent search that revisits prunning heuristic
    # TODO: prioritize edges that remove few trajectories
    assert steps in [0, 1]
    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, supports=False, bidirectional=False, ee_only=ee_only,
                                    max_directions=50, max_attempts=10, **kwargs)
    full_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                         precompute_collisions=True, supports=False, bidirectional=True, ee_only=ee_only,
                                         max_directions=250, max_attempts=1, **kwargs)
    # TODO: could just check kinematics instead of collision
    ee_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                        precompute_collisions=True, supports=False, bidirectional=True, ee_only=True,
                                        max_directions=250, max_attempts=1, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)
    # TODO: 2-step lookahead based on neighbors or spatial proximity
    # TODO: sort heuristic by most future colliding edges

    final_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, final_printed) or \
            not test_stiffness(extrusion_path, element_from_id, final_printed):
        data = {
            'sequence': None,
            'runtime': elapsed_time(start_time),
        }
        return None, data

    full_sample_traj, _ = get_sample_traj(elements, full_print_gen_fn)
    ee_sample_traj, ee_trajs_from_element = get_sample_traj(elements, ee_print_gen_fn)
    if ee_only:
        full_sample_traj = ee_sample_traj

    def sample_remaining(printed, sample_fn):
        # TODO: only check nodes that can be immediately printed?
        return all(sample_fn(printed, element) is not None for element in randomize(elements - printed))

    def conflict_fn(printed, element):
        # TODO: condition on a fixed last history to reflect the fact that we don't really want to backtrack
        # TODO: could add element if desired
        order = retrace_plan(visited, printed)
        printed = set(order[:-1])
        scores = [len(traj.colliding) for traj in ee_trajs_from_element[element]
                  if not (traj.colliding & printed)]
        # TODO: could evaluate more to get a better estimate
        # TODO: could sort by the number of trajectories need replanning if printed
        assert scores
        return -max(scores) # hardest
        #return min(scores) # easiest
        #return np.average(scores)
        #return np.random.random()

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    sample_remaining(initial_printed, ee_sample_traj)
    heuristic_fn = conflict_fn
    add_successors(queue, elements, node_points, ground_nodes, heuristic_fn, initial_printed)

    plan = None
    min_remaining = INF
    num_evaluated = 0
    worst_backtrack = 0
    num_deadends = 0
    while queue and (elapsed_time(start_time) < max_time):
        # TODO: store robot configuration
        num_evaluated += 1
        _, printed, element = heapq.heappop(queue)
        num_remaining = len(elements) - len(printed)
        backtrack = num_remaining - min_remaining
        if max_backtrack < backtrack: # backtrack_bound
            continue
        worst_backtrack = max(worst_backtrack, backtrack)
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining
        print('Iteration: {} | Best: {} | Backtrack: {} | Deadends: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, worst_backtrack, num_deadends, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        printed_nodes = compute_printed_nodes(ground_nodes, printed)
        if has_gui():
            color_structure(element_bodies, printed, element)

        next_printed = printed | {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                (stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            # Hard dead-end
            #num_deadends += 1
            continue
        #order = retrace_plan(visited, next_printed)

        # TODO: before or after sampling?
        # Constraint propagation
        # forward checking / lookahead: prove infeasibility quickly
        # https://en.wikipedia.org/wiki/Look-ahead_(backtracking)
        if (steps != 0) and not sample_remaining(next_printed, ee_sample_traj):
            # Soft dead-end
            num_deadends += 1
            #wait_for_user()
            continue

        #command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        command = full_sample_traj(printed, element)
        if command is None:
            # Soft dead-end
            #num_deadends += 1
            continue
        assert any(n in printed_nodes for n in command.trajectories[0].element)
        if command.trajectories[0].n1 not in printed_nodes:
            command = command.reverse()
        # TODO: sample several EE trajectories and then sort by non-dominated

        if not ee_only and (steps != 0) and not sample_remaining(next_printed, full_sample_traj):
            # Soft dead-end
            num_deadends += 1
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
        sequence = [traj.directed_element for traj in plan]
    data = {
        'sequence': sequence,
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_remaining,
        'num_elements': len(elements)
    }
    return plan, data