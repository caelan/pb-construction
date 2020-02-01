from __future__ import print_function

import heapq
import time
from collections import defaultdict
from termcolor import cprint

from extrusion.validator import compute_plan_deformation
from extrusion.progression import Node, retrace_trajectories, add_successors, recover_directed_sequence, \
    recover_sequence, draw_action
from extrusion.heuristics import get_heuristic_fn, score_stiffness
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn, MAX_DIRECTIONS, MAX_ATTEMPTS
from extrusion.utils import check_connected, get_id_from_element, PrintTrajectory, JOINT_WEIGHTS, compute_printed_nodes, \
    compute_printable_elements, roundrobin, get_memory_in_kb, check_memory, timeout
from extrusion.stiffness import create_stiffness_checker, test_stiffness
from extrusion.visualization import color_structure
from extrusion.motion import compute_motion, compute_motions
from extrusion.logger import export_log_data, RECORD_BT, RECORD_CONSTRAINT_VIOLATION, RECORD_QUEUE, OVERWRITE, VISUALIZE_ACTION, CHECK_BACKTRACK, QUEUE_COUNT, PAUSE_UPON_BT, MAX_STATES_STORED, RECORD_DEADEND
from extrusion.motion import compute_motion, compute_motions

from pybullet_tools.utils import INF, has_gui, elapsed_time, LockRenderer, randomize, \
    get_movable_joints, get_joint_positions, get_distance_fn, wait_for_user

# https://github.com/ContinuumIO/anaconda-issues/issues/905
import os
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
if env not in os.environ:
    os.environ[env] = '1'

##################################################

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
    cprint('#EE: {} | #Arm: {}'.format(num_ee, num_arm), 'green')

    # TODO: only check nearby remaining_elements
    # TODO: only check collisions conditioned on current decisions
    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None

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
        # ? what does num mean?
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
    else:
        cprint('Initial state check fails.', 'red')

    plan = None
    min_remaining = INF
    num_evaluated = max_backtrack = num_deadends = stiffness_failures = extrusion_failures= transit_failures = 0

    #############################################
    locker = LockRenderer()
    bt_data = []  # backtrack history
    cons_data = []  # constraint violation history
    deadend_data = []  # deadend history
    queue_data = []  # queue candidates history

    def snapshot_state(data_list=None, reason='', queue_log_cnt=0):
        """a lot of global parameters are used

        """
        cur_data = {}
        cur_data['num_evaluated'] = num_evaluated # iter
        cur_data['reason'] = reason
        cur_data['elapsed_time'] = elapsed_time(start_time)
        cur_data['min_remaining'] = min_remaining
        cur_data['max_backtrack'] = max_backtrack
        cur_data['num_deadends'] = num_deadends

        cur_data['extrusion_failures'] = extrusion_failures
        cur_data['stiffness_failures'] = stiffness_failures
        cur_data['transit_failures'] = transit_failures

        cur_data['backtrack'] = backtrack
        cur_data['total_q_len'] = len(queue)

        cur_data['chosen_element'] = element
        record_plan = retrace_trajectories(visited, printed)
        planned_elements = recover_directed_sequence(record_plan)
        cur_data['planned_elements'] = planned_elements
        cur_data['queue'] = []

        # print('++++++++++++')
        # if queue_log_cnt > 0:
        #     top_candidates = [(visits, priority, printed, element)] + list(heapq.nsmallest(queue_log_cnt, queue))
        #     for candidate in top_candidates:
        #         # * for progression
        #         temporal_chosen_element = candidate[3]
        #         temp_visits, temp_priority = candidate[0], candidate[1]
        #         temporal_structure = printed | {temporal_chosen_element}
        #         if len(temporal_structure) == len(printed):
        #             continue

        #         stiffness_score = score_stiffness(
        #             extrusion_path, element_from_id, temporal_structure, checker=checker)
        #         temp_command = sample_extrusion(
        #             print_gen_fn, ground_nodes, printed, temporal_chosen_element)
        #         extrusion_feasible = 0 if temp_command is None else 1
        #         # lower is better
        #         print('cand: {}, compl: {}, feas: {}'.format(
        #             temporal_chosen_element, stiffness_score, extrusion_feasible))
        #         cur_data['queue'].append(
        #             (list(temporal_chosen_element), stiffness_score, extrusion_feasible, temp_visits, temp_priority))
        # print('++++++++++++')

        if data_list is not None and len(data_list) < MAX_STATES_STORED:
            data_list.append(cur_data)

        if CHECK_BACKTRACK:
            draw_action(node_points, next_printed, element)
            # color_structure(element_bodies, next_printed, element)

            # TODO: can take picture here
            locker.restore()
            cprint('{} detected, press Enter to continue!'.format(reason), 'red')
            wait_for_user()
            locker = LockRenderer()
        return cur_data
    # end snapshot
    #############################################

    try:
        while queue:
            if elapsed_time(start_time) > max_time and check_memory(): #max_memory):
                if elapsed_time(start_time) < max_time:
                    cprint('memory leak: {} | {} '.format(check_memory(), get_memory_in_kb()))
                raise TimeoutError
            visits, priority, printed, element, current_conf = heapq.heappop(queue)
            num_remaining = len(all_elements) - len(printed)

            num_evaluated += 1
            print('-'*5)
            if num_remaining < min_remaining:
                min_remaining = num_remaining
                cprint('New best: {}/{}'.format(num_remaining,
                                                len(all_elements)), 'green')

            cprint('Eval Iter: {} | Best: {}/{} | Backtrack: {} | Deadends: {} | Printed: {} | Element: {} | E-Id: {} | Time: {:.3f}'.format(
                num_evaluated, min_remaining, len(all_elements), max_backtrack, num_deadends, len(printed), element, id_from_element[element], elapsed_time(start_time)))
            next_printed = printed | {element}

            backtrack = num_remaining - min_remaining
            if backtrack > max_backtrack:
                max_backtrack = backtrack
                # * (optional) visualization for diagnosis
                if RECORD_BT:
                    cprint('max backtrack increased to {}'.format(
                        max_backtrack), 'cyan')
                    snapshot_state(bt_data, reason='Backtrack')
                    if PAUSE_UPON_BT: wait_for_user()

            if backtrack_limit < backtrack:
                cprint('backtrack {} exceeds limit {}, exit.'.format(
                    backtrack, backtrack_limit), 'red')
                raise KeyboardInterrupt
                # break  # continue            

            if RECORD_QUEUE:
                snapshot_state(
                    queue_data, reason='queue_history', queue_log_cnt=QUEUE_COUNT)

            # * constraint checking and forward checking
            # ! connectivity and avoid checking duplicate states
            if next_printed in visited:
                continue
            assert check_connected(ground_nodes, next_printed)

            # ! stiffness constraint
            if stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker, verbose=False):
                cprint('&&& stiffness not passed.', 'red')
                # Hard dead-end
                #num_deadends += 1
                stiffness_failures += 1
                if RECORD_CONSTRAINT_VIOLATION:
                    snapshot_state(cons_data, reason='stiffness failure')
                continue

            # !! soft deadend checking: ee sampling
            #condition = frozenset()
            #condition = set(retrace_elements(visited, printed, horizon=2))
            #condition = printed # horizon=1
            condition = next_printed
            if not sample_remaining(condition, next_printed, ee_sample_traj, num=num_ee):
                num_deadends += 1
                cprint('$$$ An end-effector successor could not be sampled!', 'red')
                if RECORD_DEADEND:
                    snapshot_state(deadend_data, reason='ee sampling deadend')
                continue

            # ! manipulation constraint
            #command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
            command = next(iter(full_sample_traj(printed, printed, element, connected=True)), None)
            if command is None:
                # Soft dead-end
                extrusion_failures += 1
                if RECORD_CONSTRAINT_VIOLATION:
                    snapshot_state(cons_data, reason='extrusion failure')
                continue

            # !! soft deadend checking: full-traj sampling
            if not sample_remaining(condition, next_printed, full_sample_traj, num=num_arm):
                num_deadends += 1
                cprint('$$$ An full-traj successor could not be sampled!', 'red')
                if RECORD_DEADEND:
                    snapshot_state(deadend_data, reason='full-traj sampling deadend')
                continue

            start_conf = end_conf = None
            if not ee_only:
                start_conf, end_conf = command.start_conf, command.end_conf
            if (start_conf is not None) and motions and not lazy:
                motion_traj = compute_motion(robot, obstacles, element_bodies,
                                             printed, current_conf, start_conf, collisions=collisions,
                                             max_time=max_time - elapsed_time(start_time))
                if motion_traj is None:
                    cprint('>>> transition motion not passed.', 'red')
                    transit_failures += 1
                    if RECORD_CONSTRAINT_VIOLATION:
                        snapshot_state(cons_data, reason='transit_failure')
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
                        transit_failures += 1
                    else:
                        plan.append(motion_traj)
                if motions and lazy:
                    plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan,
                                           collisions=collisions, max_time=max_time - elapsed_time(start_time))
                # break
                if plan is not None:
                    break
                else:
                    # backtrack
                    transit_failures += 1
            add_successors(queue, all_elements, node_points, ground_nodes, priority_fn, next_printed, end_conf,
                           partial_orders=partial_orders)
            if revisit:
                heapq.heappush(queue, (visits + 1, priority, printed, element, current_conf))
    except (KeyboardInterrupt, TimeoutError):
        # log data
        cur_data = {}
        cur_data['algorithm'] = 'lookahead'
        cur_data['heuristic'] = heuristic
        when_stop_data = snapshot_state(reason='external stop')

        cur_data['when_stopped'] = when_stop_data
        cur_data['backtrack_history'] = bt_data
        cur_data['constraint_violation_history'] = cons_data
        cur_data['deadend_history'] = deadend_data
        cur_data['queue_history'] = queue_data

        export_log_data(extrusion_path, cur_data, overwrite=OVERWRITE, **kwargs)

        cprint('search terminated by user interruption or timeout.', 'red')
        if has_gui():
            color_structure(element_bodies, printed, element)
            locker.restore()
            wait_for_user()
        # assert False, 'search terminated.'

    max_translation, max_rotation, max_compliance = compute_plan_deformation(extrusion_path, recover_sequence(plan))
    data = {
        'sequence': recover_directed_sequence(plan),
        'runtime': elapsed_time(start_time),
        'memory': get_memory_in_kb(), # May need to update instead
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        #
        'max_translation': max_translation,
        'max_rotation': max_rotation,
        'max_compliance': max_compliance,
        #
        'stiffness_failures': stiffness_failures,
        'extrusion_failures': extrusion_failures,
        'transit_failures': transit_failures,
        #
        'backtrack_history': bt_data,
        'constraint_violation_history': cons_data,
        'deadend_history': deadend_data,
        'queue_history': queue_data,
        #
        'num_deadends': num_deadends,
    }

    if not data['sequence'] and has_gui():
        color_structure(element_bodies, printed, element)
        locker.restore()
        cprint('No plan found.', 'red')
        wait_for_user()

    return plan, data
