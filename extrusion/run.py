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
import datetime

from collections import namedtuple, OrderedDict
from itertools import product
from multiprocessing import Pool, cpu_count, TimeoutError

sys.path.append('pddlstream/')

from extrusion.motion import compute_motions, display_trajectories
from extrusion.stripstream import plan_sequence
from extrusion.utils import load_world, create_stiffness_checker, \
    downsample_nodes, check_connected, get_connected_structures, test_stiffness, evaluate_stiffness, Displacement
from extrusion.parsing import load_extrusion, draw_element, create_elements, \
    get_extrusion_path, draw_model, enumerate_paths, get_extrusion_path
from extrusion.stream import get_print_gen_fn
from extrusion.greedy import regression, progression, HEURISTICS

from pddlstream.utils import get_python_version, str_from_object
from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect, get_movable_joints, add_text, \
    get_joint_positions, LockRenderer, wait_for_user, has_gui, wait_for_duration, wait_for_interrupt, unit_pose, \
    add_line, INF, is_darwin, elapsed_time, write_pickle, user_input, reset_simulation, \
    read_pickle, get_pose, draw_pose, tform_point, Euler, Pose, multiply, remove_debug

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

def label_nodes(element_bodies, element):
    element_body = element_bodies[element]
    return [
        add_text(element[0], position=(0, 0, -0.02), parent=element_body),
        add_text(element[1], position=(0, 0, +0.02), parent=element_body),
    ]

def sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes):
    gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)
    all_trajectories = []
    for index, (element, element_body) in enumerate(element_bodies.items()):
        label_nodes(element_bodies, element)
        trajectories = []
        for node1 in element:
            for traj, in gen_fn(node1, element):
                trajectories.append(traj)
        all_trajectories.extend(trajectories)
        if not trajectories:
            return None
    return all_trajectories

##################################################

def check_plan(extrusion_path, planned_elements):
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    #checker = create_stiffness_checker(extrusion_name)

    # TODO: construct the structure in different ways (random, connected)
    handles = []
    all_connected = True
    all_stiff = True
    extruded_elements = set()
    for element in planned_elements:
        extruded_elements.add(element)
        is_connected = check_connected(ground_nodes, extruded_elements)
        structures = get_connected_structures(extruded_elements)
        is_stiff = test_stiffness(extrusion_path, element_from_id, extruded_elements)
        all_stiff &= is_stiff
        print('Elements: {} | Structures: {} | Connected: {} | Stiff: {}'.format(
            len(extruded_elements), len(structures), is_connected, is_stiff))
        is_stable = is_connected and is_stiff
        if has_gui():
            color = (0, 1, 0) if is_stable else (1, 0, 0)
            handles.append(draw_element(node_points, element, color))
            #wait_for_duration(0.5)
            if not is_stable:
                wait_for_user()
    return all_connected and all_stiff

def verify_plan(extrusion_path, planned_elements):
    # Path heuristic
    # Disable shadows
    connect(use_gui=False)
    floor, robot = load_world()
    is_valid = check_plan(extrusion_path, planned_elements)
    print('Valid:', is_valid)
    reset_simulation()
    disconnect()
    return is_valid

def test(node_points, reaction_from_node):
    handles = []
    for node in sorted(reaction_from_node):
        reactions = reaction_from_node[node]
        max_force = max(map(np.linalg.norm, reactions))
        print('node={}, max force={:.3f}'.format(node, max_force))
        print(list(map(np.array, reactions)))
        start = node_points[node]
        for reaction in reactions:
            vector = 0.05 * np.array(reaction) / max_force
            end = start + vector
            handles.append(add_line(start, end, color=(0, 1, 0)))
        wait_for_user()
        for handle in handles:
            remove_debug(handle)

def visualize_stiffness(problem, element_bodies):
    if not has_gui():
        return
    # +z points parallel to each element body
    #for element, body in element_bodies.items():
    #    print(element)
    #    label_nodes(element_bodies, element)
    #    draw_pose(get_pose(body), length=0.02)
    #    wait_for_user()

    element_from_id, node_points, ground_nodes = load_extrusion(problem)
    elements = list(element_from_id.values())
    draw_model(elements, node_points, ground_nodes)

    deformation = evaluate_stiffness(problem, element_from_id, elements)
    reaction_from_node = {}
    # Freeform Assembly Planning
    # TODO: https://arxiv.org/pdf/1801.00527.pdf
    # Though assembly sequencing is often done by finding a disassembly sequence and reversing it, we will use a forward search.
    # Thus a low-cost state will usually be correctly identified by considering only the deflection of the cantilevered beam path
    # and approximating the rest of the beams as being infinitely stiff

    # TODO: could recompute stability properties at each point
    for index, reactions in deformation.reactions.items():
        # Yijiang assumes pointing along +x
        element = element_from_id[index]
        body = element_bodies[element]
        rotation = Pose(euler=Euler(pitch=np.pi/2))
        world_from_local = multiply(rotation, get_pose(body))
        for node, reaction_local in zip(elements[index], reactions):
            # TODO: apply to torques as well
            reaction_world = tform_point(world_from_local, reaction_local[:3])
            reaction_from_node.setdefault(node, []).append(reaction_world)
    for node, reaction in deformation.fixities.items():
        reaction_from_node.setdefault(node, []).append(reaction[:3])

    #reaction_from_node = deformation.displacements # For visualizing displacements
    #test(node_points, reaction_from_node)
    total_reaction_from_node = {node: np.sum(reactions, axis=0)
                               for node, reactions in reaction_from_node.items()}
    force_from_node = {node: np.linalg.norm(reaction)
                       for node, reaction in total_reaction_from_node.items()}
    max_force = max(force_from_node.values())
    print('Max force:', max_force)
    for i, node in enumerate(sorted(total_reaction_from_node, key=lambda n: force_from_node[n])):
        print('{}) node={}, point={}, vector={}, magnitude={:.3f}'.format(
            i, node, node_points[node], total_reaction_from_node[node], force_from_node[node]))

    handles = []
    for node, reaction_world in total_reaction_from_node.items():
        start = node_points[node]
        vector = 0.05 * np.array(reaction_world) / max_force
        end = start + vector
        handles.append(add_line(start, end, color=(0, 1, 0)))
    wait_for_user()


##################################################

ALGORITHMS = [
    #'stripstream'.
    'progression',
    'regression',
]

def plan_extrusion(args, viewer=False, precompute=False, verbose=False, watch=False):
    # TODO: setCollisionFilterGroupMask
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    set_seed(hash((time.time(), args.seed)))
    # TODO: change dir for pddlstream
    element_from_id, node_points, ground_nodes = load_extrusion(args.problem, verbose=True)
    elements = list(element_from_id.values())
    #elements, ground_nodes = downsample_nodes(elements, node_points, ground_nodes)
    # plan = plan_sequence_test(node_points, elements, ground_nodes)

    connect(use_gui=viewer)
    with LockRenderer():
        draw_pose(unit_pose(), length=1.)
        floor, robot = load_world(use_floor=False)
        obstacles = [] if floor is None else [floor]
        element_bodies = dict(zip(elements, create_elements(
            node_points, elements, color=(0, 0, 0, 1))))
    # joint_weights = compute_joint_weights(robot, num=1000)
    initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    # dump_body(robot)
    visualize_stiffness(args.problem, element_bodies)
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
            planned_trajectories, data = progression(robot, obstacles, element_bodies, args.problem, heuristic=args.bias,
                                                     max_time=args.max_time, collisions=not args.cfree,
                                                     disable=args.disable, stiffness=args.stiffness)
        elif args.algorithm == 'regression':
            planned_trajectories, data = regression(robot, obstacles, element_bodies, args.problem, heuristic=args.bias,
                                                    max_time=args.max_time, collisions=not args.cfree,
                                                    disable=args.disable, stiffness=args.stiffness)
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
            planned_trajectories = compute_motions(robot, obstacles, element_bodies, initial_conf, planned_trajectories)
    reset_simulation()
    disconnect()

    # planned_elements = [traj.element for traj in planned_trajectories]
    # random.shuffle(planned_elements)
    # planned_elements = sorted(elements, key=lambda e: max(node_points[n][2] for n in e)) # TODO: tiebreak by angle or x

    #verify_plan(path, planned_elements)
    if watch:
        display_trajectories(ground_nodes, planned_trajectories)
    if not verbose:
        sys.stdout.close()
    return args, data

##################################################

Configuration = namedtuple('Configuration', ['seed', 'problem', 'algorithm', 'bias', 'max_time',
                                             'cfree', 'disable', 'stiffness', 'motions'])

def train_parallel(num=10, max_time=30*60):
    initial_time = time.time()
    print('Trials:', num)
    print('Max time:', max_time)

    problems = enumerate_paths()
    configurations = [Configuration(*c) for c in product(
        range(num), problems, ALGORITHMS, HEURISTICS, [max_time],
        [False], [False], [True], [False])]
    print('Configurations: {}'.format(len(configurations)))

    serial = is_darwin()
    available_cores = cpu_count()
    num_cores = max(1, min(1 if serial else available_cores - 3, len(configurations)))
    print('Max Cores:', available_cores)
    print('Serial:', serial)
    print('Using Cores:', num_cores)
    date = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    filename = '{}.pk{}'.format(date, get_python_version())
    path = os.path.join('experiments', filename)
    print('Data path:', path)

    user_input('Begin?')
    pool = Pool(processes=num_cores)  # , initializer=mute)
    generator = pool.imap_unordered(plan_extrusion, configurations, chunksize=1)
    results = []
    while True:
        start_time = time.time()
        try:
            configuration, data = generator.next(timeout=2 * max_time)
            print(len(results), configuration, data)
            results.append((configuration, data))
            if results:
                write_pickle(path, results)
                print('Saved', path)
        except StopIteration:
            break
        except TimeoutError:
            print('Error! Timed out after {:.3f} seconds'.format(elapsed_time(start_time)))
            break
    print('Total time:', elapsed_time(initial_time))

##################################################

Score = namedtuple('Score', ['failure', 'runtime'])

def score_result(result):
    return Score(1. - round(result['success'], 3), round(result['runtime'], 3))

def load_experiment(filename, overall=True):
    # TODO: maybe just pass the random seed as a separate arg
    # TODO: aggregate over all problems and score using IPC rules
    # https://ipc2018-classical.bitbucket.io/
    data_from_problem = OrderedDict()
    for config, result in read_pickle(filename):
        problem = 'all' if overall else config.problem
        data_from_problem.setdefault(problem, []).append((config, result))

    for p_idx, problem in enumerate(sorted(data_from_problem)):
        print()
        print('{}) Problem: {}'.format(p_idx, os.path.basename(os.path.abspath(problem))))

        data_from_config = OrderedDict()
        value_per_field = {}
        for config, result in data_from_problem[problem]:
            new_config = Configuration(None, None, *config[2:])
            #print(config._asdict()) # config.__dict__
            for field, value in config._asdict().items():
                value_per_field.setdefault(field, set()).add(value)
            data_from_config.setdefault(new_config, []).append(result)

        print('Attributes:', str_from_object(value_per_field))
        print('Configs:', len(data_from_config))
        for c_idx, config in enumerate(sorted(data_from_config, key=str)):
            results = data_from_config[config]
            accumulated_result = {}
            for result in results:
                for name, value in result.items():
                    accumulated_result.setdefault(name, []).append(value)
            mean_result = {name: round(np.average(values), 3) for name, values in accumulated_result.items()}
            score = score_result(mean_result)
            key = {field: value for field, value in config._asdict().items()
                                   if 2 <= len(value_per_field[field])}
            print('{}) {} ({}): {}'.format(c_idx, str_from_object(key), len(results), str_from_object(score)))


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
    parser.add_argument('-b', '--bias', default='z', choices=HEURISTICS,
                        help='Which heuristic to use')
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='Disables collisions with obstacles')
    parser.add_argument('-d', '--disable', action='store_true',
                        help='Disables trajectory planning')
    parser.add_argument('-m', '--motions', action='store_true',
                        help='Plans motions between each extrusion')
    parser.add_argument('-n', '--num', default=0, type=int,
                        help='TBD')
    parser.add_argument('-l', '--load', default=None,
                        help='TBD')
    parser.add_argument('-p', '--problem', default='simple_frame',
                        help='The name of the problem to solve')
    parser.add_argument('-s', '--stiffness',  action='store_false',
                        help='TBD')
    parser.add_argument('-t', '--max_time', default=30*60, type=int,
                        help='The max time')
    parser.add_argument('-v', '--viewer', action='store_true',
                        help='Enables the viewer during planning')
    args = parser.parse_args()
    print('Arguments:', args)
    np.set_printoptions(precision=3)

    if args.load is not None:
        load_experiment(args.load)
        return
    if args.num:
        train_parallel(num=args.num, max_time=args.max_time)
        return

    args.seed = hash(time.time())
    if args.problem == 'all':
        for extrusion_path in enumerate_paths():
            args.problem = extrusion_path
            plan_extrusion(args, verbose=True, watch=False)
    else:
        extrusion_path = get_extrusion_path(args.problem)
        args.problem = extrusion_path
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
# TODO: sort by deformation in the priority queue
# TODO: identify the max violating node
# TODO: compliance (work on the structure)
# TODO: introduce support structures and then require that they be removed
# Robot spiderweb printing weaving hook which may slide
# Graph traversal (path within the graph): load
# TODO: only consider axioms that could be relevant
