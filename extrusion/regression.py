import heapq
import random
import time
from termcolor import cprint

# https://github.com/ContinuumIO/anaconda-issues/issues/905
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

from extrusion.greedy import Node, sample_extrusion, retrace_trajectories, recover_sequence, recover_directed_sequence, \
    draw_action
from extrusion.heuristics import get_heuristic_fn
from extrusion.motion import compute_motion
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.utils import get_id_from_element, get_ground_elements, create_stiffness_checker, is_ground, \
    check_connected, test_stiffness
from extrusion.validator import compute_plan_deformation
from extrusion.visualization import draw_ordered
from pddlstream.utils import outgoing_from_edges
from pybullet_tools.utils import INF, get_movable_joints, get_joint_positions, randomize, implies, has_gui, \
    remove_all_debug, wait_for_user, elapsed_time, LockRenderer


def regression(robot, obstacles, element_bodies, extrusion_path, partial_orders=[],
               heuristic='z', max_time=INF, backtrack_limit=INF,
               collisions=True, stiffness=True, motions=True, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: persistent search
    # TODO: max branching factor

    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    ground_elements = get_ground_elements(all_elements, ground_nodes)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, bidirectional=False,
                                    precompute_collisions=False, max_directions=500, max_attempts=1,
                                    collisions=collisions, **kwargs)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=False)

    final_conf = initial_conf # TODO: allow choice of config
    final_printed = all_elements
    queue = []
    visited = {final_printed: Node(None, None)}

    outgoing_from_element = outgoing_from_edges(partial_orders)
    def add_successors(printed, conf):
        ground_remaining = printed <= ground_elements
        num_remaining = len(printed) - 1
        assert 0 <= num_remaining
        for element in randomize(printed):
            if outgoing_from_element[element] & printed:
                continue
            if implies(is_ground(element, ground_nodes), ground_remaining):
                bias = heuristic_fn(printed, element, conf=None)
                priority = (num_remaining, bias, random.random())
                heapq.heappush(queue, (priority, printed, element, conf))

    if check_connected(ground_nodes, final_printed) and \
            test_stiffness(extrusion_path, element_from_id, final_printed, checker=checker):
        add_successors(final_printed, final_conf)

    # TODO: lazy hill climbing using the true stiffness heuristic
    # Choose the first move that improves the score
    if has_gui():
        sequence = sorted(final_printed, key=lambda e: heuristic_fn(final_printed, e, conf=None), reverse=True)
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

    check_backtrack = False
    record_snapshots = True
    locker = LockRenderer()

    # TODO: computed number of motion planning failures
    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = 0
    stiffness_failures = transit_failures = 0
    bt_data = [] # backtrack history
    cons_data = [] # constraint violation history

    def snapshot_state(data_list=None, reason=''):
        cur_data = {}
        cur_data['iter'] = num_evaluated - 1
        cur_data['reason'] = reason
        cur_data['min_remain'] = min_remaining
        cur_data['max_backtrack'] = max_backtrack
        cur_data['backtrack'] = backtrack
        cur_data['total_q_len'] = len(queue)
        cur_data['num_stiffness_violation'] = stiffness_failures

        cur_data['chosen_element'] = element
        record_plan = retrace_trajectories(visited, printed)
        planned_elements = recover_directed_sequence(record_plan)
        cur_data['planned_elements'] = planned_elements

        # queue_log_cnt = 200
        # cur_data['queue'] = []
        # cur_data['queue'].append((id_from_element[element], priority))
        # for candidate in heapq.nsmallest(queue_log_cnt, queue):
        #     cur_data['queue'].append((id_from_element[candidate[2]], candidate[0]))

        if data_list:
            data_list.append(cur_data)
        if check_backtrack:
            draw_action(node_points, next_printed, element)
            # color_structure(element_bodies, next_printed, element)

            # TODO: can take picture here
            locker.restore()
            cprint('{} detected, press Enter to continue!'.format(reason), 'red')
            wait_for_user()
            locker = LockRenderer()
        return cur_data

    try:
        while queue and (elapsed_time(start_time) < max_time):
            priority, printed, element, current_conf = heapq.heappop(queue)
            num_remaining = len(printed)
            num_evaluated += 1
            print('-'*5)

            if num_remaining < min_remaining:
                min_remaining = num_remaining
                cprint('New best: {}/{}'.format(num_remaining, len(all_elements)), 'green')

            cprint('Eval Iter: {} | Best: {}/{} | Printed: {} | Element: {} | E-Id: {} | Time: {:.3f}'.format(
                num_evaluated, min_remaining, len(all_elements), len(printed), element, id_from_element[element], elapsed_time(start_time)))
            next_printed = printed - {element}

            backtrack = num_remaining - min_remaining
            if backtrack > max_backtrack:
                max_backtrack = backtrack
                # * (optional) visualization for diagnosis
                if record_snapshots:
                    cprint('max backtrack increased to {}'.format(max_backtrack), 'cyan')
                    snapshot_state(bt_data, reason='Backtrack')

            if backtrack_limit < backtrack:
                cprint('backtrack {} exceeds limit {}, exit.'.format(backtrack, backtrack_limit), 'red')
                break # continue

            
            # * constraint checking
            # ! connectivity and avoid checking duplicate states
            if (next_printed in visited) or not check_connected(ground_nodes, next_printed):
                continue

            # ! stiffness constraint
            if not implies(stiffness, test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
                cprint('&&& stiffness not passed.', 'red')
                stiffness_failures += 1
                # * (optional) visualization for diagnosis
                if record_snapshots:
                    snapshot_state(cons_data, reason='stiffness_violation')
                continue

            # ! extrusion feasibility
            # TODO: could do this eagerly to inspect the full branching factor
            command = sample_extrusion(print_gen_fn, ground_nodes, next_printed, element)
            if command is None:
                continue
            
            # ! transition feasibility
            if motions:
                motion_traj = compute_motion(robot, obstacles, element_bodies, node_points, printed,
                                             command.end_conf, current_conf, collisions=collisions)
                if motion_traj is None:
                    transit_failures += 1
                    cprint('>>> transition motion not passed.', 'red')
                    if record_snapshots:
                        snapshot_state(cons_data, reason='transit_failure')
                    continue
                command.trajectories.append(motion_traj)

            visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
            if not next_printed:
                min_remaining = 0
                plan = retrace_trajectories(visited, next_printed, reverse=True)
                break
            add_successors(next_printed, command.start_conf)
    except (KeyboardInterrupt, TimeoutError):
        # log data
        cur_data = {}
        cur_data['search_method'] = 'regression'
        cur_data['heuristic'] = heuristic
        when_stop_data = snapshot_state(reason='external stop')

        cur_data['when_stopped'] = when_stop_data
        cur_data['backtrack_history'] = bt_data
        cur_data['constraint_violation_history'] = cons_data

        export_log_data(extrusion_path, cur_data, overwrite=False)

        cprint('search terminated by user interruption or timeout.', 'red')
        if has_gui():
            color_structure(element_bodies, printed, element)
            locker.restore()
            wait_for_user()
        assert False, 'search terminated.'

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
        'transit_failures': transit_failures,
    }
    return plan, data