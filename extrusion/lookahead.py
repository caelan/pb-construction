from __future__ import print_function

import heapq
import time
from collections import defaultdict

from extrusion.validator import compute_plan_deformation
from extrusion.progression import Node, retrace_trajectories, add_successors, recover_directed_sequence, \
    recover_sequence
from extrusion.heuristics import get_heuristic_fn
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn, MAX_DIRECTIONS, MAX_ATTEMPTS
from extrusion.utils import check_connected, get_id_from_element, PrintTrajectory, JOINT_WEIGHTS, compute_printed_nodes, \
    compute_printable_elements, roundrobin
from extrusion.stiffness import create_stiffness_checker, test_stiffness
from extrusion.visualization import color_structure
from extrusion.motion import compute_motion, compute_motions
# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import INF, has_gui, elapsed_time, LockRenderer, randomize, \
    get_movable_joints, get_joint_positions, get_distance_fn

def retrace_elements(visited, current_state, **kwargs):
    return [traj.element for traj in retrace_trajectories(visited, current_state, **kwargs)
            if isinstance(traj, PrintTrajectory)]

##################################################

def get_sample_traj(ground_nodes, element_bodies, print_gen_fn, max_directions=INF, max_extrusions=INF,
                    condition=True, collisions=True):
    gen_from_element = {(node, element): print_gen_fn(node1=None, element=element, extruded=[], trajectories=[])
                        for element in element_bodies for node in element}
    count_per_element = {(node, element): 0 for element in element_bodies for node in element}
    trajs_from_element = defaultdict(list)
    #gen_from_element_printed = {}
    # TODO: make a generator for each parent vertex when scanning next extrusions
    # TODO: soft-heuristic that counts the number of failures but doesn't cause a hard deadend

    def enumerate_extrusions(printed, node, element):
        for traj in trajs_from_element[node, element]:
            yield traj
        if (max_directions <= count_per_element[node, element]) or \
                (max_extrusions <= len(trajs_from_element[node, element])):
            return
        with LockRenderer(True):
            if condition:
                # TODO: could perform bidirectional here again
                generator = print_gen_fn(node1=node, element=element, extruded=printed,
                                         trajectories=trajs_from_element[node, element])
            else:
                generator = gen_from_element[node, element]
            for traj in generator: # TODO: islice for the num to sample
                count_per_element[node, element] += 1
                if traj is not None:
                    traj, = traj
                    trajs_from_element[node, element].append(traj)
                    yield traj
                if max_directions <= count_per_element[node, element]:
                    print('Enumerated {}: {} directions and {} trajectories'.format(
                        element, count_per_element[node, element], len(trajs_from_element[node, element])))
                    break
            #for _ in range(100):
            #    traj, = next(print_gen_fn(None, element, extruded=[]), (None,))

    def sample_traj(printed, next_printed, element, connected=True, num=1):
        # TODO: other num conditions: max time, min collisions, etc
        assert 1 <= num
        printed_nodes = compute_printed_nodes(ground_nodes, printed)
        start_nodes = set(element) & printed_nodes if connected else element
        safe_trajectories = []
        for traj in roundrobin(*[enumerate_extrusions(printed, node, element) for node in start_nodes]):
            #safe = not (traj.colliding & next_printed)
            safe = not collisions or traj.is_safe(next_printed, element_bodies)
            if safe:
                safe_trajectories.append(traj)
            if num <= len(safe_trajectories):
                break
        return safe_trajectories

    return sample_traj, trajs_from_element

def topological_sort(robot, obstacles, element_bodies, extrusion_path):
    # TODO: take fewest collision samples and attempt to topological sort
    # Repeat if a cycle is detected
    raise NotImplementedError()

##################################################

def lookahead(robot, obstacles, element_bodies, extrusion_path, partial_orders=[], num_ee=0, num_arm=1,
              plan_all=False, use_conflicts=False, use_replan=False, heuristic='z', max_time=INF, backtrack_limit=INF,
              revisit=False, ee_only=False, collisions=True, stiffness=True, motions=True, lazy=True, **kwargs):
    if not use_conflicts:
        num_ee, num_arm = min(num_ee, 1),  min(num_arm, 1)
    if ee_only:
        num_ee, num_arm = max(num_arm, num_ee), 0
    print('#EE: {} | #Arm: {}'.format(num_ee, num_arm))
    # TODO: only check nearby remaining_elements
    # TODO: only check collisions conditioned on current decisions
    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None

    #print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
    #                                precompute_collisions=False, supports=False, ee_only=ee_only,
    #                                max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS, collisions=collisions, **kwargs)
    full_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                         precompute_collisions=False, supports=False, ee_only=ee_only, allow_failures=True,
                                         max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS, collisions=collisions, **kwargs)
    # TODO: could just check environment collisions & kinematics instead of element collisions
    ee_print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                        precompute_collisions=False, supports=False, ee_only=True, allow_failures=True,
                                        max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS, collisions=collisions, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)
    #distance_fn = get_distance_fn(robot, joints, weights=JOINT_WEIGHTS)
    # TODO: 2-step lookahead based on neighbors or spatial proximity

    full_sample_traj, full_trajs_from_element = get_sample_traj(ground_nodes, element_bodies, full_print_gen_fn,
                                                                collisions=collisions)
    ee_sample_traj, ee_trajs_from_element = get_sample_traj(ground_nodes, element_bodies, ee_print_gen_fn,
                                                            collisions=collisions)
    if ee_only:
        full_sample_traj = ee_sample_traj
    #ee_sample_traj, ee_trajs_from_element = full_sample_traj, full_trajs_from_element

    #heuristic_trajs_from_element = full_trajs_from_element if (num_ee == 0) else ee_trajs_from_element
    heuristic_trajs_from_element = full_trajs_from_element if (num_arm != 0) else ee_trajs_from_element

    #########################

    def sample_remaining(printed, next_printed, sample_fn, num=1, **kwargs):
        if num == 0:
            return True
        remaining_elements = (all_elements - next_printed) if plan_all else \
            compute_printable_elements(all_elements, ground_nodes, next_printed)
        # TODO: could just consider nodes in printed (connected=True)
        return all(sample_fn(printed, next_printed, element, connected=False, num=num, **kwargs)
                   for element in randomize(remaining_elements))

    def conflict_fn(printed, element, conf):
        # Dead-end detection without stability performs reasonably well
        # TODO: could add element if desired
        order = retrace_elements(visited, printed)
        printed = frozenset(order[:-1]) # Remove last element (to ensure at least one traj)
        if use_replan:
            remaining = list(all_elements - printed)
            requires_replan = [all(element in traj.colliding for traj in ee_trajs_from_element[e2]
                                  if not (traj.colliding & printed)) for e2 in remaining if e2 != element]
            return len(requires_replan)
        else:
            safe_trajectories = [traj for traj in heuristic_trajs_from_element[element] if not (traj.colliding & printed)]
            assert safe_trajectories
            best_traj = max(safe_trajectories, key=lambda traj: len(traj.colliding))
            num_colliding = len(best_traj.colliding)
            return -num_colliding
        #distance = distance_fn(conf, best_traj.start_conf)
        # TODO: ee distance vs conf distance
        # TODO: l0 distance based on whether we remain at the same node
        # TODO: minimize instability while printing (dynamic programming)
        #return (-num_colliding, distance)

    if use_conflicts:
        priority_fn = lambda *args: (conflict_fn(*args), heuristic_fn(*args))
    else:
        priority_fn = heuristic_fn

    #########################
    # initial search state
    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    if check_connected(ground_nodes, all_elements) and \
            test_stiffness(extrusion_path, element_from_id, all_elements) and \
            sample_remaining(initial_printed, initial_printed, ee_sample_traj, num=num_ee) and \
            sample_remaining(initial_printed, initial_printed, full_sample_traj, num=num_arm):
        add_successors(queue, all_elements, node_points, ground_nodes, priority_fn, initial_printed, initial_conf,
                       partial_orders=partial_orders)

    plan = None
    min_remaining = INF
    num_evaluated = worst_backtrack = num_deadends = stiffness_failures = extrusion_failures= transit_failures = 0
    while queue and (elapsed_time(start_time) < max_time):
        num_evaluated += 1
        visits, priority, printed, element, current_conf = heapq.heappop(queue)
        num_remaining = len(all_elements) - len(printed)
        backtrack = num_remaining - min_remaining
        worst_backtrack = max(worst_backtrack, backtrack)
        if backtrack_limit < backtrack:
            break # continue
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining
        print('Iteration: {} | Best: {} | Backtrack: {} | Deadends: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, worst_backtrack, num_deadends, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        if has_gui():
            color_structure(element_bodies, printed, element)

        next_printed = printed | {element}
        if next_printed in visited:
            continue
        # unconnected ones should be pruned by constraint propagation already
        assert check_connected(ground_nodes, next_printed)
        if stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker, verbose=False):
            # Hard dead-end
            #num_deadends += 1
            stiffness_failures += 1
            print('Partial structure is not stiff!')
            continue

        #condition = frozenset()
        #condition = set(retrace_elements(visited, printed, horizon=2))
        #condition = printed # horizon=1
        condition = next_printed

        # * soft deadend checking
        if not sample_remaining(condition, next_printed, ee_sample_traj, num=num_ee):
            # Soft dead-end
            num_deadends += 1
            #wait_for_user()
            continue

        #command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
        command = next(iter(full_sample_traj(printed, printed, element, connected=True)), None)
        if command is None:
            # Soft dead-end
            #num_deadends += 1
            print('The transition could not be sampled!')
            extrusion_failures += 1
            continue

        if not sample_remaining(condition, next_printed, full_sample_traj, num=num_arm):
            # Soft dead-end
            num_deadends += 1
            continue

        start_conf = end_conf = None
        if not ee_only:
            start_conf, end_conf = command.start_conf, command.end_conf
        if (start_conf is not None) and motions:
            motion_traj = compute_motion(robot, obstacles, element_bodies,
                                         printed, current_conf, start_conf, collisions=collisions,
                                         max_time=max_time - elapsed_time(start_time))
            if motion_traj is None:
                transit_failures += 1
                continue
            command.trajectories.insert(0, motion_traj)

        visited[next_printed] = Node(command, printed)
        if all_elements <= next_printed:
            # TODO: anytime mode
            min_remaining = 0
            plan = retrace_trajectories(visited, next_printed)
            if motions and not lazy:
                motion_traj = compute_motion(robot, obstacles, element_bodies, frozenset(),
                                             initial_conf, plan[0].start_conf, collisions=collisions,
                                             max_time=max_time - elapsed_time(start_time))
                if motion_traj is None:
                    plan = None
                else:
                    plan.append(motion_traj)
            if motions and lazy:
                plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan,
                                       collisions=collisions, max_time=max_time - elapsed_time(start_time))
            if plan is not None:
                break
            else:
                transit_failures += 1
        add_successors(queue, all_elements, node_points, ground_nodes, priority_fn, next_printed, end_conf,
                       partial_orders=partial_orders)
        if revisit:
            heapq.heappush(queue, (visits + 1, priority, printed, element, current_conf))

    max_translation, max_rotation = compute_plan_deformation(extrusion_path, recover_sequence(plan))
    data = {
        'sequence': recover_directed_sequence(plan),
        'runtime': elapsed_time(start_time),
        'num_elements': len(all_elements),
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': worst_backtrack,
        'max_translation': max_translation,
        'max_rotation': max_rotation,
        'stiffness_failures': stiffness_failures,
        'extrusion_failures': extrusion_failures,
        'transit_failures': transit_failures,
    }
    return plan, data
