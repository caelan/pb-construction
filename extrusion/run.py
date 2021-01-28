#!/usr/bin/env python
from __future__ import print_function

import sys
import argparse
import cProfile
import pstats
import numpy as np
import time
import os
import json

sys.path.extend([
    'pddlstream/',
    'ss-pybullet/',
])

from extrusion.figure import DEFAULT_MAX_TIME
from extrusion.visualization import label_element, set_extrusion_camera, label_nodes, display_trajectories, \
    BACKGROUND_COLOR, draw_model, SHADOWS, draw_ordered
from extrusion.experiment import train_parallel
from extrusion.motion import compute_motions, validate_trajectories
from extrusion.stripstream import stripstream
from extrusion.utils import load_world, TOOL_LINK, compute_sequence_distance, get_print_distance, \
    recover_sequence, recover_directed_sequence, Profiler, extract_plan_data
from extrusion.parsing import load_extrusion, create_elements_bodies, \
    enumerate_problems, get_extrusion_path, affine_extrusion, parse_origin_pose
from extrusion.stream import get_print_gen_fn
from extrusion.progression import progression, get_global_parameters
from extrusion.regression import regression
from extrusion.heuristics import HEURISTICS
from extrusion.validator import check_plan, compute_plan_deformation
from extrusion.lookahead import lookahead
from extrusion.stiffness import plan_stiffness, create_stiffness_checker

from pybullet_tools.transformations import scale_matrix, rotation_matrix
from pybullet_tools.utils import connect, disconnect, get_movable_joints, get_joint_positions, LockRenderer, \
    unit_pose, reset_simulation, draw_pose, apply_alpha, BLACK, Pose, Euler, has_gui, set_numpy_seed, \
    set_random_seed, INF, link_from_name, get_link_pose, point_from_pose, WorldSaver, elapsed_time, \
    timeout, get_configuration, RED, wait_if_gui, apply_affine, invert, multiply, tform_from_pose, write_json, read_json

DEFAULT_SCALE = 1.

SCALE_ASSEMBLY = {
    'simple_frame': 5, # 3.5 | 5
    'topopt-101_tiny': 3.5, # 2. | 3.5
    'four-frame': 5.
}

##################################################

def sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes):
    gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)
    all_trajectories = []
    for index, (element, element_body) in enumerate(element_bodies.items()):
        label_element(element_bodies, element)
        trajectories = []
        for node1 in element:
            for traj, in gen_fn(node1, element):
                trajectories.append(traj)
        all_trajectories.extend(trajectories)
        if not trajectories:
            return None
    return all_trajectories

##################################################

def get_base_centroid(node_points, ground_nodes):
    # TODO: could do this just for the ground nodes
    centroid = np.average(node_points, axis=0)
    min_z = np.min(node_points, axis=0)[2]  # - 1e-2
    return np.append(centroid[:2], [min_z])

# Introduce support scaffolding fixities and then require that they be removed
# Robot spiderweb printing weaving hook which may slide
def scale_assembly(elements, node_points, ground_nodes, scale=1.):
    if scale is None:
        return node_points
    base_centroid = get_base_centroid(node_points, ground_nodes)
    scaled_node_points = [scale*(point - base_centroid) + base_centroid
                          for point in node_points] # + np.array([0., 0., 2e-2])
    #draw_model(elements, scaled_node_points, ground_nodes, color=RED)
    #wait_if_gui()
    return scaled_node_points

def rotate_assembly(elements, node_points, ground_nodes, yaw=0.):
    if yaw is None:
        return node_points
    # TODO: more general affine transformations
    world_from_base = Pose(point=get_base_centroid(node_points, ground_nodes))
    points_base = apply_affine(invert(world_from_base), node_points)
    rotation = Pose(euler=Euler(yaw=yaw))
    points_world = list(map(np.array, apply_affine(multiply(world_from_base, rotation), points_base)))
    #draw_model(elements, scaled_node_points, ground_nodes, color=RED)
    #wait_if_gui()
    return points_world

def transform_assembly(problem, elements, node_points, ground_nodes):
    # TODO: need to write the new structure to ensure the same points are used downstream
    #elements = downsample_structure(elements, node_points, ground_nodes, num=None)
    #elements, ground_nodes = downsample_nodes(elements, node_points, ground_nodes)

    scale = SCALE_ASSEMBLY.get(problem, DEFAULT_SCALE)
    if scale is not None:
        node_points = scale_assembly(elements, node_points, ground_nodes, scale)

    #yaw = np.pi / 8
    yaw = None
    if yaw is not None:
        node_points = rotate_assembly(elements, node_points, ground_nodes, yaw)
    return node_points

##################################################

def transform_json(problem):
    original_path = get_extrusion_path(problem)
    #new_path = '{}_transformed.json'.format(problem)
    new_path = 'transformed.json'.format(problem) # TODO: create folder

    #pose = Pose(euler=Euler(roll=np.pi / 2.))
    yaw = np.pi / 4.
    pose = Pose(euler=Euler(yaw=yaw))
    tform = tform_from_pose(pose)
    tform = rotation_matrix(angle=yaw, direction=[0, 0, 1])

    scale = SCALE_ASSEMBLY.get(problem, DEFAULT_SCALE)
    #tform = scale*np.identity(4)
    tform = scale_matrix(scale, origin=None, direction=None)

    json_data = affine_extrusion(original_path, tform)
    write_json(new_path, json_data)
    # TODO: rotate the whole robot as well
    # TODO: could also use the z heuristic when upside down
    return new_path

##################################################

def solve_extrusion(robot, obstacles, element_from_id, node_points, element_bodies, extrusion_path, ground_nodes, args,
                    partial_orders=[], precompute=False, **kwargs):
    # TODO: could treat ground as partial orders
    backtrack_limit = INF # 0 | INF
    seed = hash((time.time(), args.seed))
    set_numpy_seed(seed)
    set_random_seed(seed)

    # initial_position = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK)))
    # elements = sorted(element_bodies.keys())
    # sequence = plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, elements,
    #                           initial_position=initial_position, max_time=INF, max_backtrack=INF)
    # if has_gui():
    #     draw_ordered(sequence, node_points)
    #     wait_if_gui()
    # return

    initial_conf = get_configuration(robot)
    if args.algorithm == 'stripstream':
        sampled_trajectories = []
        if precompute:
            sampled_trajectories = sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes)
        plan, data = stripstream(robot, obstacles, node_points, element_bodies, ground_nodes,
                                 trajectories=sampled_trajectories, collisions=not args.cfree,
                                 max_time=args.max_time, disable=args.disable, **kwargs)
    elif args.algorithm == 'progression':
        plan, data = progression(robot, obstacles, element_bodies, extrusion_path, partial_orders=partial_orders,
                                 heuristic=args.bias, max_time=args.max_time,
                                 backtrack_limit=backtrack_limit, collisions=not args.cfree,
                                 disable=args.disable, stiffness=args.stiffness, motions=args.motions, **kwargs)
    elif args.algorithm == 'regression':
        plan, data = regression(robot, obstacles, element_bodies, extrusion_path, partial_orders=partial_orders,
                                heuristic=args.bias, max_time=args.max_time,
                                backtrack_limit=backtrack_limit, collisions=not args.cfree,
                                disable=args.disable, stiffness=args.stiffness, motions=args.motions, **kwargs)
    elif args.algorithm == 'lookahead':
        plan, data = lookahead(robot, obstacles, element_bodies, extrusion_path,
                               partial_orders=partial_orders, heuristic=args.bias,
                               max_time=args.max_time, backtrack_limit=backtrack_limit,
                               ee_only=args.ee_only, collisions=not args.cfree,
                               disable=args.disable, stiffness=args.stiffness, motions=args.motions, **kwargs)
    else:
        raise ValueError(args.algorithm)
    if args.motions:
        plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan, collisions=not args.cfree)
    return plan, data

def plan_extrusion(args_list, viewer=False, verify=False, verbose=False, watch=False):
    results = []
    if not args_list:
        return results
    # TODO: setCollisionFilterGroupMask
    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    problems = {args.problem for args in args_list}
    assert len(problems) == 1
    [problem] = problems

    # TODO: change dir for pddlstream
    extrusion_path = get_extrusion_path(problem)
    if False:
        extrusion_path = transform_json(problem)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path, verbose=True)
    elements = sorted(element_from_id.values())
    #node_points = transform_model(problem, elements, node_points, ground_nodes)

    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with LockRenderer(lock=True):
        draw_pose(unit_pose(), length=1.)
        json_data = read_json(extrusion_path)
        draw_pose(parse_origin_pose(json_data))
        draw_model(elements, node_points, ground_nodes)

        obstacles, robot = load_world()
        color = apply_alpha(BLACK, alpha=0) # 0, 1
        #color = None
        element_bodies = dict(zip(elements, create_elements_bodies(node_points, elements, color=color)))
        set_extrusion_camera(node_points)
        #if viewer:
        #    label_nodes(node_points)
        saver = WorldSaver()
    wait_if_gui()

    #visualize_stiffness(extrusion_path)
    #debug_elements(robot, node_points, node_order, elements)
    initial_position = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK)))

    checker = None
    plan = None
    for args in args_list:
        if args.stiffness and (checker is None):
            checker = create_stiffness_checker(extrusion_path, verbose=False)

        saver.restore()
        #initial_conf = get_joint_positions(robot, get_movable_joints(robot))
        with LockRenderer(lock=not viewer):
            start_time = time.time()
            plan, data = None, {}
            with timeout(args.max_time):
                with Profiler(num=10, cumulative=False):
                    plan, data = solve_extrusion(robot, obstacles, element_from_id, node_points, element_bodies,
                                                 extrusion_path, ground_nodes, args, checker=checker)
            runtime = elapsed_time(start_time)

        sequence = recover_directed_sequence(plan)
        if verify:
            max_translation, max_rotation = compute_plan_deformation(extrusion_path, recover_sequence(plan))
            valid = check_plan(extrusion_path, sequence)
            print('Valid:', valid)
            safe = validate_trajectories(element_bodies, obstacles, plan)
            print('Safe:', safe)
            data.update({
                'safe': safe,
                'valid': valid,
                'max_translation': max_translation,
                'max_rotation': max_rotation,
            })

        data.update({
            'runtime': runtime,
            'num_elements': len(elements),
            'ee_distance': compute_sequence_distance(node_points, sequence, start=initial_position, end=initial_position),
            'print_distance': get_print_distance(plan, teleport=True),
            'distance': get_print_distance(plan, teleport=False),
            'sequence': sequence,
            'parameters': get_global_parameters(),
            'problem':  args.problem,
            'algorithm': args.algorithm,
            'heuristic': args.bias,
            'plan_extrusions': not args.disable,
            'use_collisions': not args.cfree,
            'use_stiffness': args.stiffness,
        })
        print(data)
        results.append((args, data))

        if True:
            data.update({
                'assembly': read_json(extrusion_path),
                'plan': extract_plan_data(plan), # plan | trajectories
            })
            plan_file = '{}_solution.json'.format(args.problem)
            plan_path = os.path.join('solutions', plan_file)
            write_json(plan_path, data)

    reset_simulation()
    disconnect()
    if watch and (plan is not None):
        args = args_list[-1]
        animate = not (args.disable or args.ee_only)
        connect(use_gui=True, shadows=SHADOWS, color=BACKGROUND_COLOR) # TODO: avoid reconnecting
        obstacles, robot = load_world()
        display_trajectories(node_points, ground_nodes, plan, #time_step=None, video=True,
                             animate=animate)
        reset_simulation()
        disconnect()

    if not verbose:
        sys.stdout.close()
    return results

##################################################

def create_parser():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='regression',
                        help='Which algorithm to use')
    parser.add_argument('-b', '--bias', default='plan-stiffness', choices=HEURISTICS,
                        help='Which heuristic to use')
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='Disables collisions with obstacles')
    parser.add_argument('-d', '--disable', action='store_true',
                        help='Disables trajectory planning')
    parser.add_argument('-e', '--ee_only', action='store_true',
                        help='Disables arm planning')
    parser.add_argument('-m', '--motions', action='store_false',
                        help='Plans motions between each extrusion')
    parser.add_argument('-s', '--stiffness',  action='store_false',
                        help='Disables stiffness checking')
    parser.add_argument('-t', '--max_time', default=DEFAULT_MAX_TIME, type=int,
                        help='The max time')
    return parser

def main():
    parser = create_parser()
    parser.add_argument('-p', '--problem', default='simple_frame',
                        help='The name of the problem to solve')
    parser.add_argument('-v', '--viewer', action='store_true',
                        help='Enables the viewer during planning')
    args = parser.parse_args()
    if args.disable:
        args.cfree = True
        args.motions = False
    args.seed = hash(time.time())
    print('Arguments:', args)

    # if args.problem == 'all':
    #     for problem in enumerate_problems():
    #         args.problem = problem
    #         plan_extrusion([args], verbose=True, watch=False)
    # else:
    plan_extrusion([args], viewer=args.viewer, verbose=True, watch=True)


if __name__ == '__main__':
    main()
