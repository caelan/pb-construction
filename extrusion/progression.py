from __future__ import print_function

import heapq
import random
import time

from collections import namedtuple

from extrusion.validator import compute_plan_deformation
from extrusion.heuristics import get_heuristic_fn
from pybullet_tools.utils import elapsed_time, \
    LockRenderer, reset_simulation, disconnect, randomize
from extrusion.parsing import load_extrusion
from extrusion.visualization import draw_element
from extrusion.stream import get_print_gen_fn, STEP_SIZE, APPROACH_DISTANCE, MAX_DIRECTIONS, MAX_ATTEMPTS
from extrusion.utils import check_connected, get_id_from_element, load_world, PrintTrajectory, \
    compute_printed_nodes, compute_printable_elements, RESOLUTION, timeout
from extrusion.stiffness import TRANS_TOL, ROT_TOL, create_stiffness_checker, test_stiffness
from extrusion.motion import compute_motion, compute_motions

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
from pybullet_tools.utils import connect, ClientSaver, wait_for_user, INF, has_gui, remove_all_debug, \
    get_movable_joints, get_joint_positions
from pddlstream.utils import incoming_from_edges

#State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])


##################################################

def get_global_parameters():
    return {
        'translation_tolerance': TRANS_TOL,
        'rotation_tolerance': ROT_TOL,
        'joint_resolution': RESOLUTION,
        'step_size': STEP_SIZE,
        'approach_distance': APPROACH_DISTANCE,
        'max_directions': MAX_DIRECTIONS,
        'max_attempts': MAX_ATTEMPTS,
    }

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
    for node in element:
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

def add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, printed, conf,
                   partial_orders=[], visualize=False):
    incoming_from_element = incoming_from_edges(partial_orders)
    remaining = all_elements - printed
    num_remaining = len(remaining) - 1
    #assert 0 <= num_remaining
    bias_from_element = {}
    for element in randomize(compute_printable_elements(all_elements, ground_nodes, printed)):
        if not (incoming_from_element[element] <= printed):
            continue
        bias = heuristic_fn(printed, element, conf)
        priority = (num_remaining, bias, random.random())
        visits = 0
        heapq.heappush(queue, (visits, priority, printed, element, conf))
        bias_from_element[element] = bias

    # if visualize and has_gui():
    #     handles = []
    #     with LockRenderer():
    #         remove_all_debug()
    #         for element in printed:
    #             handles.append(draw_element(node_points, element, color=(0, 0, 0)))
    #         successors = sorted(bias_from_element, key=lambda e: bias_from_element[e])
    #         handles.extend(draw_ordered(successors, node_points))
    #     print('Min: {:.3E} | Max: {:.3E}'.format(bias_from_element[successors[0]], bias_from_element[successors[-1]]))
    #     wait_for_user()

def progression(robot, obstacles, element_bodies, extrusion_path, partial_orders=[],
                heuristic='z', max_time=INF, backtrack_limit=INF,
                stiffness=True, motions=True, collisions=True, lazy=True, **kwargs):

    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, precompute_collisions=False,
                                    max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS,
                                    collisions=collisions, **kwargs)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=True)

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed: Node(None, None)}
    if check_connected(ground_nodes, all_elements) and \
            test_stiffness(extrusion_path, element_from_id, all_elements):
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, initial_printed, initial_conf,
                       partial_orders=partial_orders)

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = stiffness_failures = extrusion_failures = transit_failures = 0
    with timeout(max_time):
        while queue and (elapsed_time(start_time) < max_time):
            num_evaluated += 1
            visits, _, printed, element, current_conf = heapq.heappop(queue)
            num_remaining = len(all_elements) - len(printed)
            backtrack = num_remaining - min_remaining
            max_backtrack = max(max_backtrack, backtrack)
            if backtrack_limit < backtrack:
                break # continue
            num_evaluated += 1
            if num_remaining < min_remaining:
                min_remaining = num_remaining
            print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
                num_evaluated, min_remaining, len(printed), element, id_from_element[element], elapsed_time(start_time)))
            next_printed = printed | {element}
            assert check_connected(ground_nodes, next_printed)
            if (next_printed in visited) or (stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
                stiffness_failures += 1
                continue
            command = sample_extrusion(print_gen_fn, ground_nodes, printed, element)
            if command is None:
                extrusion_failures += 1
                continue
            if motions and not lazy:
                # TODO: test reachability from initial_conf
                motion_traj = compute_motion(robot, obstacles, element_bodies, printed,
                                             current_conf, command.start_conf, collisions=collisions,
                                             max_time=max_time - elapsed_time(start_time))
                if motion_traj is None:
                    transit_failures += 1
                    continue
                command.trajectories.insert(0, motion_traj)

            visited[next_printed] = Node(command, printed)
            if all_elements <= next_printed:
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
                break
                # if plan is not None:
                #     break
            add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn, next_printed, command.end_conf,
                           partial_orders=partial_orders)

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
        'stiffness_failures': stiffness_failures,
        'extrusion_failures': extrusion_failures,
        'transit_failures': transit_failures,
    }
    return plan, data
