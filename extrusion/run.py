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
from extrusion.stripstream import plan_sequence
from extrusion.utils import load_world, TOOL_LINK
from extrusion.parsing import load_extrusion, create_elements_bodies, \
    enumerate_problems, get_extrusion_path, affine_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.progression import progression, recover_directed_sequence, get_global_parameters
from extrusion.regression import regression
from extrusion.heuristics import HEURISTICS
from extrusion.validator import verify_plan
from extrusion.lookahead import lookahead
from extrusion.stiffness import plan_stiffness

from pybullet_tools.utils import connect, disconnect, get_movable_joints, get_joint_positions, LockRenderer, \
    unit_pose, reset_simulation, draw_pose, apply_alpha, BLACK, Pose, Euler, set_numpy_seed, \
    set_random_seed, INF, wait_for_user, link_from_name, get_link_pose, point_from_pose


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

def rotate_problem(problem_path, roll=np.pi):
    tform = Pose(euler=Euler(roll=roll))
    json_data = affine_extrusion(problem_path, tform)
    path = 'rotated.json' # TODO: create folder
    with open(path, 'w') as f:
        json.dump(json_data, f, indent=2, sort_keys=True)
    problem_path = path
    # TODO: rotate the whole robot as well
    # TODO: could also use the z heuristic when upside down
    return problem_path

##################################################

def plan_extrusion(args, viewer=False, precompute=False, verify=False, verbose=False, watch=False):
    # TODO: setCollisionFilterGroupMask
    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    seed = hash((time.time(), args.seed))
    set_numpy_seed(seed)
    set_random_seed(seed)
    # TODO: change dir for pddlstream
    problem_path = get_extrusion_path(args.problem)
    #problem_path = rotate_problem(problem_path)

    element_from_id, node_points, ground_nodes = load_extrusion(problem_path, verbose=True)
    elements = list(element_from_id.values())
    #elements = downsample_structure(elements, node_points, ground_nodes, num=None)
    #elements, ground_nodes = downsample_nodes(elements, node_points, ground_nodes)
    # plan = plan_sequence_test(node_points, elements, ground_nodes)

    partial_orders = [] # TODO: could treat ground as partial orders
    backtrack_limit = INF # 0 | INF

    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR) # TODO: avoid reconnecting
    with LockRenderer(False):
        #draw_pose(unit_pose(), length=1.)
        obstacles, robot = load_world()
        #draw_model(elements, node_points, ground_nodes)
        #wait_for_user()
        color = apply_alpha(BLACK, alpha=1) # 0, 1
        #color = None
        element_bodies = dict(zip(elements, create_elements_bodies(
            node_points, elements, color=color)))
        set_extrusion_camera(node_points)
        if viewer:
            label_nodes(node_points)

    # joint_weights = compute_joint_weights(robot, num=1000)

    initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    initial_position = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK)))
    # dump_body(robot)
    #visualize_stiffness(problem_path)
    # debug_elements(robot, node_points, node_order, elements)
    # sequence = plan_stiffness(problem_path, element_from_id, node_points, ground_nodes, elements,
    #                           initial_position=initial_position, max_time=INF, max_backtrack=INF)
    # if viewer:
    #     draw_ordered(sequence, node_points)
    #     wait_for_user()
    # return

    with LockRenderer(True):
        # TODO: reuse checker for multiple trials to account for memory leaks
        pr = cProfile.Profile()
        pr.enable()
        if args.algorithm == 'stripstream':
            sampled_trajectories = []
            if precompute:
                sampled_trajectories = sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes)
            trajectories, data = plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                                               trajectories=sampled_trajectories, collisions=not args.cfree,
                                               max_time=args.max_time, disable=args.disable, debug=False)
        elif args.algorithm == 'progression':
            trajectories, data = progression(robot, obstacles, element_bodies, problem_path, partial_orders=partial_orders,
                                             heuristic=args.bias, max_time=args.max_time,
                                             backtrack_limit=backtrack_limit, collisions=not args.cfree,
                                             disable=args.disable, stiffness=args.stiffness, motions=args.motions)
        elif args.algorithm == 'regression':
            trajectories, data = regression(robot, obstacles, element_bodies, problem_path,
                                            heuristic=args.bias, max_time=args.max_time,
                                            backtrack_limit=backtrack_limit, collisions=not args.cfree,
                                            disable=args.disable, stiffness=args.stiffness, motions=args.motions)
        elif args.algorithm == 'lookahead':
            trajectories, data = lookahead(robot, obstacles, element_bodies, problem_path,
                                           partial_orders=partial_orders, heuristic=args.bias,
                                           max_time=args.max_time, backtrack_limit=backtrack_limit,
                                           ee_only=args.ee_only, collisions=not args.cfree,
                                           disable=args.disable, stiffness=args.stiffness, motions=args.motions)
        else:
            raise ValueError(args.algorithm)
        pr.disable()
        pstats.Stats(pr).sort_stats('tottime').print_stats(10) # tottime | cumtime
        print(data)
        if trajectories is None:
            if not verbose:
                sys.stdout.close()
            return args, data
        if args.motions:
            trajectories = compute_motions(robot, obstacles, element_bodies, initial_conf,
                                           trajectories, collisions=not args.cfree)

    safe = validate_trajectories(element_bodies, obstacles, trajectories) if verify else True
    data['safe'] = safe
    if verify:
        print('Safe:', safe)
    reset_simulation()
    disconnect()

    #id_from_element = get_id_from_element(element_from_id)
    #planned_ids = [id_from_element[traj.element] for traj in trajectories]
    sequence = recover_directed_sequence(trajectories)
    valid = verify_plan(problem_path, sequence) if verify else True #, use_gui=not animate)
    data.update({
        'safe': safe,
        'valid': valid,
        'parameters': get_global_parameters(),
    })

    plan_data = {
        'problem':  args.problem,
        'algorithm': args.algorithm,
        'heuristic': args.bias,
        'plan_extrusions': not args.disable,
        'use_collisions': not args.cfree,
        'use_stiffness': args.stiffness,
        'plan': sequence,
        'safe': safe,
        'valid': valid,
    }
    plan_data.update(data)
    del plan_data['sequence']
    #plan_path = '{}_solution.json'.format(args.problem)
    #with open(plan_path, 'w') as f:
    #    json.dump(plan_data, f, indent=2, sort_keys=True)

    if watch:
        animate = not (args.disable or args.ee_only)
        display_trajectories(node_points, ground_nodes, trajectories, #time_step=None, video=True,
                             animate=animate)
    if not verbose:
        sys.stdout.close()
    return args, data

##################################################

def main():
    parser = argparse.ArgumentParser()
    # simple_frame | Nodes: 12 | Ground: 4 | Elements: 19
    # topopt-100 | Nodes: 88 | Ground: 20 | Elements: 132
    # topopt-205 | Nodes: 89 | Ground: 19 | Elements: 164
    # mars-bubble | Nodes: 97 | Ground: 11 | Elements: 225
    # djmm_test_block | Nodes: 76 | Ground: 13 | Elements: 253
    # voronoi | Nodes: 162 | Ground: 14 | Elements: 306
    # topopt-310 | Nodes: 160 | Ground: 39 | Elements: 310
    # sig_artopt-bunny | Nodes: 219 | Ground: 14 | Elements: 418
    # djmm_bridge | Nodes: 1548 | Ground: 258 | Elements: 6427
    # djmm_test_block | Nodes: 76 | Ground: 13 | Elements: 253
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
    parser.add_argument('-n', '--num', default=0, type=int,
                        help='Number of experiment trials')
    parser.add_argument('-p', '--problem', default='simple_frame',
                        help='The name of the problem to solve')
    parser.add_argument('-s', '--stiffness',  action='store_false',
                        help='Disables stiffness checking')
    parser.add_argument('-t', '--max_time', default=DEFAULT_MAX_TIME, type=int,
                        help='The max time')
    parser.add_argument('-v', '--viewer', action='store_true',
                        help='Enables the viewer during planning')
    args = parser.parse_args()
    if args.disable:
        args.cfree = True
        args.motions = False
        args.max_time = 5*60 # 259 for duck
    print('Arguments:', args)
    np.set_printoptions(precision=3)

    if args.num:
        train_parallel(args)
        return

    args.seed = hash(time.time())
    if args.problem == 'all':
        for problem in enumerate_problems():
            args.problem = problem
            plan_extrusion(args, verbose=True, watch=False)
    else:
        plan_extrusion(args, viewer=args.viewer, verbose=True, watch=True)

    # python -m extrusion.run -n 10 2>&1 | tee log.txt


if __name__ == '__main__':
    main()

# TODO: local search to reduce the violation
# TODO: introduce support structure fixities and then require that they be removed
# Robot spiderweb printing weaving hook which may slide
