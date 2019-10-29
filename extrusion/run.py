#!/usr/bin/env python
from __future__ import print_function

import sys
import argparse
import cProfile
import pstats
import numpy as np
import random
import time
import os
import json

sys.path.extend([
    'pddlstream/',
    'ss-pybullet/',
])

from extrusion.visualization import label_element, set_extrusion_camera, label_nodes
from extrusion.experiment import load_experiment, train_parallel
from extrusion.motion import compute_motions, display_trajectories
from extrusion.stripstream import plan_sequence
from extrusion.utils import load_world, get_id_from_element, PrintTrajectory
from extrusion.parsing import load_extrusion, create_elements_bodies, \
    enumerate_problems, get_extrusion_path
from extrusion.stream import get_print_gen_fn
from extrusion.greedy import regression, progression, GREEDY_HEURISTICS, GREEDY_ALGORITHMS
from extrusion.validator import verify_plan
from extrusion.deadend import lookahead

from pybullet_tools.utils import connect, disconnect, get_movable_joints, get_joint_positions, LockRenderer, \
    unit_pose, reset_simulation, draw_pose, apply_alpha, BLACK

# TODO: sort by action cost heuristic
# http://www.fast-downward.org/Doc/Evaluator#Max_evaluator

ALGORITHMS = GREEDY_ALGORITHMS #+ [STRIPSTREAM_ALGORITHM]

##################################################

def get_random_seed():
    # random.getstate()[1][0]
    return np.random.get_state()[1][0]

def set_seed(seed):
    # These generators are different and independent
    random.seed(seed)
    np.random.seed(seed % (2**32))
    print('Seed:', seed)

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

def plan_extrusion(args, viewer=False, precompute=False, verbose=False, watch=False):
    # TODO: setCollisionFilterGroupMask
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    set_seed(hash((time.time(), args.seed)))
    # TODO: change dir for pddlstream
    problem_path = get_extrusion_path(args.problem)

    # #roll = 0
    # roll = np.pi
    # tform = Pose(euler=Euler(roll=roll))
    # json_data = affine_extrusion(problem_path, tform)
    # path = 'rotated.json' # TODO: folder
    # with open(path, 'w') as f:
    #     json.dump(json_data, f, indent=2, sort_keys=True)
    # problem_path = path
    # TODO: rotate the whole robot as well
    # TODO: could also use the z heuristic when upside down

    # TODO: lazily plan for the end-effector before each manipulation
    element_from_id, node_points, ground_nodes = load_extrusion(problem_path, verbose=True)
    elements = list(element_from_id.values())
    #elements, ground_nodes = downsample_nodes(elements, node_points, ground_nodes)
    # plan = plan_sequence_test(node_points, elements, ground_nodes)

    connect(use_gui=viewer)
    with LockRenderer():
        draw_pose(unit_pose(), length=1.)
        obstacles, robot = load_world()
        alpha = 1 # 0
        element_bodies = dict(zip(elements, create_elements_bodies(
            node_points, elements, color=apply_alpha(BLACK, alpha))))
        set_extrusion_camera(node_points)
        if viewer:
            label_nodes(node_points)

    # joint_weights = compute_joint_weights(robot, num=1000)
    initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    # dump_body(robot)
    #visualize_stiffness(problem_path)
    # debug_elements(robot, node_points, node_order, elements)

    with LockRenderer(False):
        trajectories = []
        if precompute:
            trajectories = sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes)
        pr = cProfile.Profile()
        pr.enable()
        if args.algorithm == 'stripstream':
            planned_trajectories, data = plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                                                       trajectories=trajectories, collisions=not args.cfree,
                                                       max_time=args.max_time, disable=args.disable, debug=False)
        elif args.algorithm == 'progression':
            planned_trajectories, data = progression(robot, obstacles, element_bodies, problem_path, heuristic=args.bias,
                                                     max_time=args.max_time, collisions=not args.cfree,
                                                     disable=args.disable, stiffness=args.stiffness)
        elif args.algorithm == 'regression':
            planned_trajectories, data = regression(robot, obstacles, element_bodies, problem_path, heuristic=args.bias,
                                                    max_time=args.max_time, collisions=not args.cfree,
                                                    disable=args.disable, stiffness=args.stiffness)
        elif args.algorithm == 'deadend':
            planned_trajectories, data = lookahead(robot, obstacles, element_bodies, problem_path, heuristic=args.bias,
                                                   max_time=args.max_time, ee_only=args.ee_only, collisions=not args.cfree,
                                                   disable=args.disable, stiffness=args.stiffness, motions=args.motions)
        else:
            raise ValueError(args.algorithm)
        pr.disable()
        pstats.Stats(pr).sort_stats('tottime').print_stats(10) # tottime | cumtime
        print(data)
        if planned_trajectories is None:
            if not verbose:
                sys.stdout.close()
            return args, data
        if args.motions:
            planned_trajectories = compute_motions(robot, obstacles, element_bodies, initial_conf, planned_trajectories,
                                                   collisions=not args.cfree)
    reset_simulation()
    disconnect()

    #id_from_element = get_id_from_element(element_from_id)
    #planned_ids = [id_from_element[traj.element] for traj in planned_trajectories]
    planned_elements = [traj.directed_element for traj in planned_trajectories
                        if isinstance(traj, PrintTrajectory)]
    # random.shuffle(planned_elements)
    # planned_elements = sorted(elements, key=lambda e: max(node_points[n][2] for n in e)) # TODO: tiebreak by angle or x
    animate = not (args.disable or args.ee_only)
    valid = verify_plan(problem_path, planned_elements) #, use_gui=not animate)

    plan_data = {
        'problem':  args.problem,
        'algorithm': args.algorithm,
        'heuristic': args.bias,
        'plan_extrusions': not args.disable,
        'use_collisions': not args.cfree,
        'use_stiffness': args.stiffness,
        'plan': planned_elements,
        'valid': valid,
    }
    plan_data.update(data)
    del plan_data['sequence']
    plan_path = '{}_solution.json'.format(args.problem)
    with open(plan_path, 'w') as f:
        json.dump(plan_data, f, indent=2, sort_keys=True)

    if watch:
        display_trajectories(node_points, ground_nodes, planned_trajectories, animate=animate)
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
    parser.add_argument('-b', '--bias', default='z', choices=GREEDY_HEURISTICS,
                        help='Which heuristic to use')
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='Disables collisions with obstacles')
    parser.add_argument('-d', '--disable', action='store_true',
                        help='Disables trajectory planning')
    parser.add_argument('-e', '--ee_only', action='store_true',
                        help='Disables arm planning')
    parser.add_argument('-m', '--motions', action='store_true',
                        help='Plans motions between each extrusion')
    parser.add_argument('-n', '--num', default=0, type=int,
                        help='TBD')
    parser.add_argument('-l', '--load', default=None,
                        help='Analyze an experiment')
    parser.add_argument('-p', '--problem', default='simple_frame',
                        help='The name of the problem to solve')
    parser.add_argument('-s', '--stiffness',  action='store_false',
                        help='Disables stiffness checking')
    parser.add_argument('-t', '--max_time', default=30*60, type=int,
                        help='The max time')
    parser.add_argument('-v', '--viewer', action='store_true',
                        help='Enables the viewer during planning')
    args = parser.parse_args()
    print('Arguments:', args)
    np.set_printoptions(precision=3)

    # TODO: randomly rotate the structure and analyze the effectiveness of different heuristics
    # The stiffness checker assumes gravity is always -z. Apply rotation to heuristic

    if args.load is not None:
        load_experiment(args.load)
        return
    if args.num:
        train_parallel(num=args.num, max_time=args.max_time)
        return

    args.seed = hash(time.time())
    if args.problem == 'all':
        for problem in enumerate_problems():
            args.problem = problem
            plan_extrusion(args, verbose=True, watch=False)
    else:
        plan_extrusion(args, viewer=args.viewer, verbose=True, watch=True)

    # TODO: collisions at the ends of elements?
    # TODO: slow down automatically near endpoints
    # TODO: heuristic that orders elements by angle
    # TODO: check that both the start and end satisfy
    # TODO: return to start when done

    # Can greedily print
    # four-frame, simple_frame, voronoi

    # Cannot greedily print
    # topopt-100
    # mars_bubble
    # djmm_bridge
    # djmm_test_block

    # python -m extrusion.run -n 10 2>&1 | tee log.txt


if __name__ == '__main__':
    main()

# TODO: look at the actual violation of the stiffness
# TODO: local search to reduce the violation
# TODO: introduce support structures and then require that they be removed
# Robot spiderweb printing weaving hook which may slide
# Graph traversal (path within the graph): load
# TODO: only consider axioms that could be relevant
