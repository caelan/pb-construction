#!/usr/bin/env python
from __future__ import print_function

import sys
import argparse

sys.path.append('pddlstream/')

from extrusion.motion import compute_motions, display_trajectories
from extrusion.sorted import heuristic_planner
from extrusion.stripstream import plan_sequence
from extrusion.utils import load_world, create_stiffness_checker, \
    downsample_nodes, check_connected, get_connected_structures, check_stiffness
from extrusion.parsing import load_extrusion, draw_element, create_elements, \
    get_extrusion_path, draw_model, enumerate_paths, get_extrusion_path
from extrusion.stream import get_print_gen_fn
from extrusion.greedy import regression, progression

from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect, get_movable_joints, add_text, \
    get_joint_positions, LockRenderer, wait_for_user, has_gui, wait_for_duration, wait_for_interrupt

from pddlstream.utils import INF


def sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes):
    gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)
    all_trajectories = []
    for index, (element, element_body) in enumerate(element_bodies.items()):
        add_text(element[0], position=(0, 0, -0.05), parent=element_body)
        add_text(element[1], position=(0, 0, +0.05), parent=element_body)
        trajectories = []
        for node1 in element:
            for traj, in gen_fn(node1, element):
                trajectories.append(traj)
        all_trajectories.extend(trajectories)
        if not trajectories:
            return None
    return all_trajectories

##################################################

def check_plan(extrusion_name, planned_elements):
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_name)
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
        is_stiff = check_stiffness(extrusion_name, element_from_id, extruded_elements)
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


##################################################

def plan_extrusion(path, args, precompute=False):
    # TODO: setCollisionFilterGroupMask
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    element_from_id, node_points, ground_nodes = load_extrusion(path)
    elements = list(element_from_id.values())
    elements, ground_nodes = downsample_nodes(elements, node_points, ground_nodes)

    # plan = plan_sequence_test(node_points, elements, ground_nodes)
    # elements = elements[:50] # 10 | 50 | 100 | 150

    connect(use_gui=args.viewer)
    with LockRenderer():
        floor, robot = load_world()
        obstacles = [floor]
        element_bodies = dict(zip(elements, create_elements(
            node_points, elements, color=(0, 0, 0, 0))))
    # joint_weights = compute_joint_weights(robot, num=1000)
    initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    # dump_body(robot)
    if has_gui():
        draw_model(elements, node_points, ground_nodes)
        wait_for_user()
    # debug_elements(robot, node_points, node_order, elements)

    # TODO: script to solve all of them and report results
    with LockRenderer(False):
        trajectories = []
        if precompute:
            trajectories = sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes)
        if args.algorithm == 'stripstream':
            planned_trajectories = plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                                                 trajectories=trajectories, collisions=not args.cfree,
                                                 disable=args.disable, max_time=args.max_time)
        elif args.algorithm == 'regression':
            planned_trajectories = regression(robot, obstacles, element_bodies, path,
                                              collisions=not args.cfree, disable=args.disable)
        elif args.algorithm == 'progression':
            planned_trajectories = progression(robot, obstacles, element_bodies, path,
                                               collisions=not args.cfree, disable=args.disable)
        elif args.algorithm == 'heuristic':
            planned_trajectories = heuristic_planner(robot, obstacles, node_points, element_bodies, ground_nodes,
                                                     collisions=not args.cfree, disable=args.disable)
        else:
            raise ValueError(args.algorithm)
        if planned_trajectories is None:
            return
        planned_elements = [traj.element for traj in planned_trajectories]
        if args.motions:
            planned_trajectories = compute_motions(robot, obstacles, element_bodies, initial_conf, planned_trajectories)
    disconnect()

    # random.shuffle(planned_elements)
    # planned_elements = sorted(elements, key=lambda e: max(node_points[n][2] for n in e)) # TODO: tiebreak by angle or x

    # Path heuristic
    # Disable shadows
    connect(use_gui=False)
    floor, robot = load_world()
    is_valid = check_plan(path, planned_elements)
    print('Valid:', is_valid)
    if args.disable:
        wait_for_user()
        return
    disconnect()

    display_trajectories(ground_nodes, planned_trajectories)
    disconnect()

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
    parser.add_argument('-a', '--algorithm', default='stripstream',
                        help='Which algorithm to use')
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='Disables collisions with obstacles')
    parser.add_argument('-d', '--disable', action='store_true',
                        help='Disables trajectory planning')
    parser.add_argument('-m', '--motions', action='store_true',
                        help='Plans motions between each extrusion')
    parser.add_argument('-p', '--problem', default='simple_frame',
                        help='The name of the problem to solve')
    parser.add_argument('-t', '--max_time', default=INF, type=int,
                        help='The max time')
    parser.add_argument('-v', '--viewer', action='store_true',
                        help='Enables the viewer during planning (slow!)')
    args = parser.parse_args()
    print('Arguments:', args)

    if args.problem == 'all':
        for path in enumerate_paths():
            plan_extrusion(path, args)
    else:
        path = get_extrusion_path(args.problem)
        plan_extrusion(path, args)

    # TODO: collisions at the ends of elements?
    # TODO: slow down automatically near endpoints
    # TODO: heuristic that orders elements by angle
    # TODO: check that both teh start and end satisfy
    # TODO: return to start when done

    # Can greedily print
    # four-frame, simple_frame, voronoi

    # Cannot greedily print
    # topopt-100
    # mars_bubble
    # djmm_bridge
    # djmm_test_block


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
