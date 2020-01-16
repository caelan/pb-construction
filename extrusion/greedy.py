from __future__ import print_function

# https://github.com/ContinuumIO/anaconda-issues/issues/905
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'

import heapq
import random
import time
from termcolor import cprint

from collections import namedtuple

from extrusion.validator import compute_plan_deformation
from extrusion.heuristics import get_heuristic_fn
from pybullet_tools.utils import elapsed_time, \
    LockRenderer, reset_simulation, disconnect, randomize
from extrusion.parsing import load_extrusion, ELEMENT_DIAMETER, ELEMENT_SHRINK
from extrusion.visualization import draw_element, draw_ordered, draw_model, color_structure
from extrusion.stream import get_print_gen_fn
from extrusion.utils import check_connected, test_stiffness, \
    create_stiffness_checker, get_id_from_element, load_world, PrintTrajectory, \
    compute_printed_nodes, compute_printable_elements
from extrusion.motion import compute_motion
from extrusion.stream import MAX_DIRECTIONS, MAX_ATTEMPTS, STEP_SIZE

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import connect, ClientSaver, wait_for_user, INF, has_gui, remove_all_debug, \
    get_movable_joints, get_joint_positions
from pddlstream.utils import incoming_from_edges

#State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])

##################################################

def retrace_trajectories(visited, current_state, horizon=INF, reverse=False):
    command, prev_state = visited[current_state]
    if (prev_state is None) or (horizon == 0):
        return []
    prior_trajectories = retrace_trajectories(visited, prev_state, horizon=horizon-1, reverse=reverse)
    current_trajectories = [traj for traj in command.trajectories]
    if reverse:
        return current_trajectories + prior_trajectories
    return prior_trajectories + current_trajectories
    # TODO: search over local stability for each node

def recover_sequence(plan):
    if plan is None:
        return plan
    return [traj.element for traj in plan if isinstance(traj, PrintTrajectory)]

def recover_directed_sequence(plan):
    if plan is None:
        return plan
    return [traj.directed_element for traj in plan if isinstance(traj, PrintTrajectory)]

##################################################

def sample_extrusion(print_gen_fn, ground_nodes, printed, element):
    printed_nodes = compute_printed_nodes(ground_nodes, printed)
    # TODO: could always reverse these trajectories
    for node in element:
        # try generating path starting from each of the nodes
        # TODO: sample between different orientations if both are feasible
        if node in printed_nodes:
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
        cprint('Failure!', 'red')
        wait_for_user()
        reset_simulation()
        disconnect()

def draw_action(node_points, printed, element, remained_elements=None):
    """ draw printed elements in green and currently selected element in red, remained in blue
    """
    if not has_gui():
        return []
    with LockRenderer():
        remove_all_debug()
        handles = [draw_element(node_points, element, color=(1, 0, 0))]
        handles.extend(draw_element(node_points, e, color=(0, 1, 0)) for e in printed)
        if remained_elements:
            handles.extend(draw_element(node_points, e, color=(0, 0, 1)) for e in remained_elements)
    # wait_for_user()
    return handles

##################################################    

def export_log_data(extrusion_file_path, log_data, overwrite=True, indent=None):
    import os
    import datetime
    import json
    from collections import OrderedDict

    with open(extrusion_file_path, 'r') as f:
        shape_data = json.loads(f.read())
    
    if 'model_name' in shape_data:
        file_name = shape_data['model_name']
    else:
        file_name = extrusion_file_path.split('.json')[-2].split(os.sep)[-1]

    # result_file_dir = r'C:\Users\yijiangh\Documents\pb_ws\pychoreo\tests\test_data'
    here = os.path.abspath(os.path.dirname(__file__))
    result_file_dir = here
    result_file_dir = os.path.join(result_file_dir, 'extrusion_log')
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir) 
    
    data = OrderedDict()
    data['file_name'] = file_name
    date = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    data['write_time'] = date
    data.update(log_data)

    file_name_tag = log_data['search_method'] + '-' + log_data['heuristic']
    # if log_data['heuristic'] in ['stiffness', 'fixed-stiffness']:
    #     file_name_tag += '-' + log_data['stiffness_criteria']
    plan_path = os.path.join(result_file_dir, '{}_log_{}{}.json'.format(file_name, 
        file_name_tag,  '_'+data['write_time'] if not overwrite else ''))
    with open(plan_path, 'w') as f:
        json.dump(data, f, indent=indent)

##################################################

def add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, printed, conf,
                   partial_orders=[], visualize=False):
    """add successor for the progression search algorithm
    """
    incoming_from_element = incoming_from_edges(partial_orders)
    remaining = all_elements - printed
    num_remaining = len(remaining) - 1
    assert 0 <= num_remaining
    bias_from_element = {}
    for element in randomize(compute_printable_elements(all_elements, ground_nodes, printed)):
        if not (incoming_from_element[element] <= printed):
            continue
        bias = heuristic_fn(printed, element, conf)
        priority = (num_remaining, bias, random.random())
        visits = 0
        heapq.heappush(queue, (visits, priority, printed, element, conf))
        bias_from_element[element] = bias

    if visualize and has_gui():
        handles = []
        with LockRenderer():
            remove_all_debug()
            for element in printed:
                handles.append(draw_element(node_points, element, color=(0, 0, 0)))
            successors = sorted(bias_from_element, key=lambda e: bias_from_element[e])
            values = ['{:.3E}'.format(bias_from_element[e]) for e in successors]
            handles.extend(draw_ordered(successors, node_points, values))
        print('Bias Min: {:.3E} | Max: {:.3E}'.format(bias_from_element[successors[0]], bias_from_element[successors[-1]]))
        wait_for_user()

def progression(robot, obstacles, element_bodies, extrusion_path, partial_orders=[],
                heuristic='z', max_time=INF, backtrack_limit=INF,
                stiffness=True, motions=True, collisions=True, **kwargs):

    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, bidirectional=False,
                                    precompute_collisions=False, max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS,
                                    collisions=collisions, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)

    # ! step-by-step diagnosis
    visualize_action = False
    check_backtrack = False
    record_snapshots = True

    initial_printed = frozenset()
    queue = []
    # next printed state: (current command, current printed state)
    visited = {initial_printed: Node(None, None)}
    if check_connected(ground_nodes, all_elements) and \
            test_stiffness(extrusion_path, element_from_id, all_elements):
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, initial_printed, initial_conf,
                       partial_orders=partial_orders, visualize=visualize_action)
    else:
        cprint('Grounded nodes are not connected to any of the elements or the whole structure is not stiff!', 'red')

    locker = LockRenderer()
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
        while queue:
            if elapsed_time(start_time) > max_time:
                raise TimeoutError
            visits, _, printed, element, current_conf = heapq.heappop(queue)
            num_remaining = len(all_elements) - len(printed)
            num_evaluated += 1
            print('-'*5)
            if num_remaining < min_remaining:
                min_remaining = num_remaining
                cprint('New best: {}/{}'.format(num_remaining, len(all_elements)), 'green')
            cprint('Eval Iter: {} | Best: {}/{} | Printed: {} | Element: {} | E-Id: {} | Time: {:.3f}'.format(
                num_evaluated, min_remaining, len(all_elements), len(printed), element, id_from_element[element], elapsed_time(start_time)))
            next_printed = printed | {element}

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
            # connectivity and avoid checking duplicate states
            if (next_printed in visited) or not check_connected(ground_nodes, next_printed):
                continue
            # stiffness constraint
            if stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker):
                cprint('&&& stiffness not passed.', 'red')
                stiffness_failures += 1
                # * (optional) visualization for diagnosis
                if record_snapshots:
                    snapshot_state(cons_data, reason='stiffness_violation')
                continue
            # manipulation constraint
            command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
            if command is None:
                continue
            # transition motion constraint
            if motions:
                motion_traj = compute_motion(robot, obstacles, element_bodies, node_points, printed,
                                             current_conf, command.start_conf, collisions=collisions)
                if motion_traj is None:
                    cprint('>>> transition motion not passed.', 'red')
                    transit_failures += 1
                    if record_snapshots:
                        snapshot_state(cons_data, reason='transit_failure')
                    continue
                command.trajectories.insert(0, motion_traj)

            visited[next_printed] = Node(command, printed)
            if all_elements <= next_printed:
                min_remaining = 0
                plan = retrace_trajectories(visited, next_printed)
                break
            add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, next_printed, command.end_conf,
                           partial_orders=partial_orders, visualize=visualize_action)
    except (KeyboardInterrupt, TimeoutError):
        # log data
        cur_data = {}
        cur_data['search_method'] = 'progression'
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
        'transit_failures' : transit_failures,
        'stiffness_failures' : stiffness_failures,
    }

    if not data['sequence'] and has_gui():
        color_structure(element_bodies, printed, element)
        locker.restore()
        wait_for_user()

    return plan, data
