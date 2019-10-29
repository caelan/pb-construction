from __future__ import print_function

import heapq
import time
import numpy as np
from collections import defaultdict

from extrusion.greedy import get_heuristic_fn, Node, retrace_trajectories, retrace_elements, add_successors, compute_printed_nodes
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, get_id_from_element, PrintTrajectory
from extrusion.visualization import color_structure
from extrusion.motion import compute_motion, JOINT_WEIGHTS
# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import INF, has_gui, elapsed_time, LockRenderer, randomize, \
    wait_for_user, get_movable_joints, get_joint_positions, get_distance_fn


def get_sample_traj(elements, print_gen_fn):
    gen_from_element = {element: print_gen_fn(None, element, extruded=[], trajectories=[]) for element in elements}
    trajs_from_element = defaultdict(list)
    # TODO: option to make additional samples (only useful for sorting heuristic)

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

def topological_sort(robot, obstacles, element_bodies, extrusion_path):
    # TODO: take fewest collision samples and attempt to topological sort
    # Repeat if a cycle is detected
    raise NotImplementedError()

def lookahead(robot, obstacles, element_bodies, extrusion_path,
              heuristic='z', max_time=INF, max_backtrack=INF,
              ee_only=False, collisions=True, stiffness=True, motions=True, steps=1, **kwargs):
    # TODO: persistent search that revisits prunning heuristic
    # TODO: prioritize edges that remove few trajectories
    assert steps in [0, 1]
    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, supports=False, bidirectional=False, ee_only=ee_only,
                                    max_directions=50, max_attempts=10, collisions=collisions, **kwargs)
    full_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                         precompute_collisions=True, supports=False, bidirectional=True, ee_only=ee_only,
                                         max_directions=250, max_attempts=1, collisions=collisions, **kwargs)
    # TODO: could just check kinematics instead of collision
    ee_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                        precompute_collisions=True, supports=False, bidirectional=True, ee_only=True,
                                        max_directions=250, max_attempts=1, collisions=collisions, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)
    # TODO: 2-step lookahead based on neighbors or spatial proximity

    final_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, final_printed) or \
            not test_stiffness(extrusion_path, element_from_id, final_printed):
        data = {
            'sequence': None,
            'runtime': elapsed_time(start_time),
        }
        return None, data

    full_sample_traj, full_trajs_from_element = get_sample_traj(all_elements, full_print_gen_fn)
    ee_sample_traj, ee_trajs_from_element = get_sample_traj(all_elements, ee_print_gen_fn)
    if ee_only:
        full_sample_traj = ee_sample_traj
    ee_sample_traj, ee_trajs_from_element = full_sample_traj, full_trajs_from_element

    def sample_remaining(printed, sample_fn):
        # TODO: only check nodes that can be printed given the current nodes?
        # TODO: condition on a fixed last history to reflect the fact that we don't really want to backtrack
        return all(sample_fn(printed, element) is not None
                   for element in randomize(all_elements - printed))

    distance_fn = get_distance_fn(robot, joints, weights=JOINT_WEIGHTS)
    def conflict_fn(printed, element, conf):
        # TODO: could add element if desired
        #return np.random.random()
        #return 0 # Dead-end detection without stability performs well
        order = retrace_elements(visited, printed)
        printed = frozenset(order[:-1]) # Remove last element (to ensure at least one traj)
        #remaining = list(all_elements - printed)
        #requires_replan = [all(element in traj.colliding for traj in ee_trajs_from_element[e2]
        #                       if not (traj.colliding & printed)) for e2 in remaining if e2 != element]
        #return len(requires_replan)

        # TODO: could evaluate more to get a better estimate
        safe_trajectories = [traj for traj in ee_trajs_from_element[element] if not (traj.colliding & printed)]
        assert safe_trajectories
        best_traj = max(safe_trajectories, key=lambda traj: len(traj.colliding))
        num_colliding = len(best_traj.colliding)
        return -num_colliding
        #distance = distance_fn(conf, best_traj.start_conf)
        # TODO: ee distance vs conf distance
        #return (-num_colliding, distance)

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    sample_remaining(initial_printed, ee_sample_traj)
    #priority_fn = heuristic_fn
    #priority_fn = conflict_fn
    priority_fn = lambda p, e, q: (conflict_fn(p, e, q), heuristic_fn(p, e))
    add_successors(queue, all_elements, node_points, ground_nodes, priority_fn, initial_printed, initial_conf)

    plan = None
    min_remaining = INF
    num_evaluated = worst_backtrack = num_deadends = 0
    while queue and (elapsed_time(start_time) < max_time):
        num_evaluated += 1
        priority, printed, element, current_conf = heapq.heappop(queue)
        num_remaining = len(all_elements) - len(printed)
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
        #order = retrace_elements(visited, next_printed)

        # TODO: locally optimize solution by conditioning on discrete decisions
        # TODO: anytime algorithm
        # TODO: minimize instability while printing (dynamic programming)

        # TODO: the directionality actually matters for the printing orientation
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

        start_conf = end_conf = None
        if not ee_only:
            start_conf, end_conf = command.start_conf, command.end_conf
        if (start_conf is not None) and motions:
            motion_traj = compute_motion(robot, obstacles, element_bodies,
                                         printed, current_conf, start_conf, collisions=collisions)
            if motion_traj is None:
                continue
            command.trajectories.insert(0, motion_traj)

        # TODO: separate parameters for dead-end versus transition
        visited[next_printed] = Node(command, printed)
        if all_elements <= next_printed:
            min_remaining = 0
            plan = retrace_trajectories(visited, next_printed)
            break
        add_successors(queue, all_elements, node_points, ground_nodes, priority_fn, next_printed, end_conf)

    sequence = None
    if plan is not None:
        sequence = [traj.directed_element for traj in plan if isinstance(traj, PrintTrajectory)]
    data = {
        'sequence': sequence,
        'runtime': elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'num_remaining': min_remaining,
        'num_elements': len(all_elements)
    }
    return plan, data