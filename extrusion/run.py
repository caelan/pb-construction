#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append('pddlstream/')

import cProfile
import pstats
import numpy as np
import argparse

from extrusion.extrusion_utils import create_elements, \
    load_extrusion, TOOL_NAME, load_world, \
    get_node_neighbors, draw_element, get_disabled_collisions, MotionTrajectory, PrintTrajectory, is_ground, \
    get_supported_orders, element_supports, is_start_node, doubly_printable, retrace_supporters
from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect, wait_for_interrupt, \
    get_movable_joints, set_joint_positions, link_from_name, add_line, get_link_pose, wait_for_duration, add_text, \
    plan_joint_motion, \
    point_from_pose, get_joint_positions, LockRenderer
from extrusion.stream import get_print_gen_fn, get_wild_print_gen_fn, test_stiffness, SELF_COLLISIONS


from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import PDDLProblem, And, print_solution
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.utils import read, get_file_path, user_input, neighbors_from_orders


JOINT_WEIGHTS = [0.3078557810844393, 0.443600199302506, 0.23544367607317915,
                 0.03637161028426032, 0.04644626184081511, 0.015054267683041092]


##################################################

def get_pddlstream(robot, obstacles, node_points, element_bodies, ground_nodes,
                   trajectories=[], collisions=True):
    # TODO: instantiation slowness is due to condition effects
    # Regression works well here because of the fixed goal state
    # TODO: plan for the end-effector first

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    constant_map = {}

    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-print': get_wild_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                              collisions=collisions),
        'test-stiffness': from_test(test_stiffness),
    }

    # TODO: assert that all elements have some support
    init = []
    for n in ground_nodes:
        init.append(('Grounded', n))
    for e in element_bodies:
        for n in e:
            if element_supports(e, n, node_points):
                init.append(('Supports', e, n))
            if is_start_node(n, e, node_points):
                init.append(('StartNode', n, e))
    for e in element_bodies:
        n1, n2 = e
        init.extend([
            ('Node', n1),
            ('Node', n2),
            ('Element', e),
            ('Printed', e),
            ('Edge', n1, e, n2),
            ('Edge', n2, e, n1),
            #('StartNode', n1, e),
            #('StartNode', n2, e),
        ])
        #if is_ground(e, ground_nodes):
        #    init.append(('Grounded', e))
    #for e1, neighbors in get_element_neighbors(element_bodies).items():
    #    for e2 in neighbors:
    #        init.append(('Supports', e1, e2))
    for t in trajectories:
        init.extend([
            ('Traj', t),
            ('PrintAction', t.n1, t.element, t),
        ])

    goal = And(*[('Removed', e) for e in element_bodies])

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                  trajectories=[], collisions=True,
                  debug=True, max_time=30):
    if trajectories is None:
        return None
    # TODO: iterated search using random restarts
    # TODO: most of the time seems to be spent extracting the stream plan
    # TODO: NEGATIVE_SUFFIX to make axioms easier
    pr = cProfile.Profile()
    pr.enable()
    pddlstream_problem = get_pddlstream(robot, obstacles, node_points, element_bodies,
                                        ground_nodes, trajectories=trajectories, collisions=collisions)
    #solution = solve_incremental(pddlstream_problem, planner='add-random-lazy', max_time=600,
    #                             max_planner_time=300, debug=True)
    stream_info = {
        'sample-print': StreamInfo(PartialInputs(unique=True)),
    }
    #planner = 'ff-ehc'
    planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    solution = solve_focused(pddlstream_problem, stream_info=stream_info, max_time=max_time,
                             effort_weight=1, unit_efforts=True, max_skeletons=None, unit_costs=True, bind=False,
                             planner=planner, max_planner_time=15, debug=debug, reorder=False)
    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    print_solution(solution)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)
    plan, _, _ = solution
    if plan is None:
        return None
    return [t for _, (n1, e, t) in reversed(plan)]

##################################################

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

def compute_motions(robot, fixed_obstacles, element_bodies, initial_conf, trajectories):
    # TODO: can just plan to initial and then shortcut
    # TODO: backoff motion
    # TODO: reoptimize for the sequence that have the smallest movements given this
    # TODO: sample trajectories
    # TODO: more appropriate distance based on displacement/volume
    if trajectories is None:
        return None
    weights = np.array(JOINT_WEIGHTS)
    resolutions = np.divide(0.005*np.ones(weights.shape), weights)
    movable_joints = get_movable_joints(robot)
    disabled_collisions = get_disabled_collisions(robot)
    printed_elements = []
    current_conf = initial_conf
    all_trajectories = []
    for i, print_traj in enumerate(trajectories):
        set_joint_positions(robot, movable_joints, current_conf)
        goal_conf = print_traj.path[0]
        obstacles = fixed_obstacles + [element_bodies[e] for e in printed_elements]
        path = plan_joint_motion(robot, movable_joints, goal_conf, obstacles=obstacles,
                                 self_collisions=SELF_COLLISIONS, disabled_collisions=disabled_collisions,
                                 weights=weights, resolutions=resolutions,
                                 restarts=5, iterations=50, smooth=100)
        if path is None:
            print('Failed to find a motion plan!')
            return None
        motion_traj = MotionTrajectory(robot, movable_joints, path)
        print('{}) {}'.format(i, motion_traj))
        all_trajectories.append(motion_traj)
        current_conf = print_traj.path[-1]
        printed_elements.append(print_traj.element)
        all_trajectories.append(print_traj)
    # TODO: return to initial?
    return all_trajectories

##################################################

def display_trajectories(ground_nodes, trajectories, time_step=0.05):
    if trajectories is None:
        return
    connect(use_gui=True)
    floor, robot = load_world()
    wait_for_interrupt()
    movable_joints = get_movable_joints(robot)
    #element_bodies = dict(zip(elements, create_elements(node_points, elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    connected = set(ground_nodes)
    for trajectory in trajectories:
        if isinstance(trajectory, PrintTrajectory):
            print(trajectory, trajectory.n1 in connected, trajectory.n2 in connected,
                  is_ground(trajectory.element, ground_nodes), len(trajectory.path))
            connected.add(trajectory.n2)
        #wait_for_interrupt()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []
        for conf in trajectory.path:
            set_joint_positions(robot, movable_joints, conf)
            if isinstance(trajectory, PrintTrajectory):
                current_point = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_NAME)))
                if last_point is not None:
                    color = (0, 0, 1) if is_ground(trajectory.element, ground_nodes) else (1, 0, 0)
                    handles.append(add_line(last_point, current_point, color=color))
                last_point = current_point
            wait_for_duration(time_step)
        #wait_for_interrupt()
    #user_input('Finish?')
    wait_for_interrupt()
    disconnect()

##################################################

def debug_elements(robot, node_points, node_order, elements):
    #test_grasps(robot, node_points, elements)
    #test_print(robot, node_points, elements)
    #return

    for element in elements:
       color = (0, 0, 1) if doubly_printable(element, node_points) else (1, 0, 0)
       draw_element(node_points, element, color=color)
    wait_for_interrupt('Continue?')

    # TODO: topological sort
    node = node_order[40]
    node_neighbors = get_node_neighbors(elements)
    for element in node_neighbors[node]:
       color = (0, 1, 0) if element_supports(element, node, node_points) else (1, 0, 0)
       draw_element(node_points, element, color)

    element = elements[-1]
    draw_element(node_points, element, (0, 1, 0))
    incoming_edges, _ = neighbors_from_orders(get_supported_orders(elements, node_points))
    supporters = []
    retrace_supporters(element, incoming_edges, supporters)
    for e in supporters:
       draw_element(node_points, e, (1, 0, 0))
    wait_for_interrupt('Continue?')

    #for name, args in plan:
    #   n1, e, n2 = args
    #   draw_element(node_points, e)
    #   user_input('Continue?')
    #test_ik(robot, node_order, node_points)

##################################################

def main(precompute=False):
    parser = argparse.ArgumentParser()
    # djmm_test_block | mars_bubble | sig_artopt-bunny | topopt-100 | topopt-205 | topopt-310 | voronoi
    parser.add_argument('-p', '--problem', default='simple_frame', help='The name of the problem to solve')
    parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions with obstacles')
    parser.add_argument('-m', '--motions', action='store_true', help='Plans motions between each extrusion')
    parser.add_argument('-t', '--max_time', default=10*60, type=int, help='The max time')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    args = parser.parse_args()
    print('Arguments:', args)

    # TODO: setCollisionFilterGroupMask
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    # TODO: submodule in https://github.mit.edu/yijiangh/assembly-instances

    elements, node_points, ground_nodes = load_extrusion(args.problem)
    node_order = list(range(len(node_points)))
    #np.random.shuffle(node_order)
    node_order = sorted(node_order, key=lambda n: node_points[n][2])
    elements = sorted(elements, key=lambda e: min(node_points[n][2] for n in e))

    #node_order = node_order[:100]
    ground_nodes = [n for n in ground_nodes if n in node_order]
    elements = [element for element in elements if all(n in node_order for n in element)]
    #plan = plan_sequence_test(node_points, elements, ground_nodes)

    connect(use_gui=args.viewer)
    floor, robot = load_world()
    obstacles = [floor]
    initial_conf = get_joint_positions(robot, get_movable_joints(robot))
    #dump_body(robot)
    #if has_gui():
    #    draw_model(elements, node_points, ground_nodes)
    #    wait_for_interrupt('Continue?')

    #joint_weights = compute_joint_weights(robot, num=1000)
    #elements = elements[:50] # 10 | 50 | 100 | 150
    #debug_elements(robot, node_points, node_order, elements)
    element_bodies = dict(zip(elements, create_elements(node_points, elements)))

    with LockRenderer():
        trajectories = []
        if precompute:
            trajectories = sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes)
        plan = plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                             trajectories=trajectories, collisions=not args.cfree, max_time=args.max_time)
        if args.motions:
            plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan)

    disconnect()
    display_trajectories(ground_nodes, plan)
    # TODO: collisions at the ends of elements?


if __name__ == '__main__':
    main()

"""
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  2568280  158.181    0.000  158.181    0.000 {pybullet.getClosestPoints}
 13586760   42.924    0.000   78.596    0.000 pddlstream/pddlstream/algorithms/scheduling/recover_axioms.py:132(is_useful_atom)
     2844   37.479    0.013   46.247    0.016 pddlstream/pddlstream/algorithms/scheduling/recover_axioms.py:88(get_achieving_axioms)
     7107   16.440    0.002   16.440    0.002 {method 'read' of 'file' objects}
 13100059   14.246    0.000   15.782    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:226(__init__)
  9785439   12.672    0.000   38.681    0.000 pddlstream/pddlstream/algorithms/downward.py:365(literal_holds)
  9780687   11.766    0.000   63.793    0.000 pddlstream/pddlstream/algorithms/scheduling/plan_streams.py:110(<lambda>)
 55357785   11.722    0.000   11.722    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:14(__hash__)
 19561374   10.463    0.000   49.126    0.000 pddlstream/pddlstream/algorithms/scheduling/plan_streams.py:110(<genexpr>)
 10959470   10.140    0.000   10.140    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:230(__eq__)
"""