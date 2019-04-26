#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append('pddlstream/')

import cProfile
import pstats
import numpy as np
import argparse
import time

from extrusion.extrusion_utils import TOOL_NAME, load_world, \
    get_node_neighbors, get_disabled_collisions, MotionTrajectory, PrintTrajectory, is_ground, \
    get_supported_orders, element_supports, is_start_node, doubly_printable, retrace_supporters
from extrusion.parsing import load_extrusion, draw_element, create_elements
from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect, wait_for_interrupt, \
    get_movable_joints, set_joint_positions, link_from_name, add_line, get_link_pose, wait_for_duration, add_text, \
    plan_joint_motion, point_from_pose, get_joint_positions, LockRenderer
from extrusion.stream import get_print_gen_fn, get_wild_print_gen_fn, test_stiffness, SELF_COLLISIONS


from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import PDDLProblem, And, print_solution
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.utils import read, get_file_path, user_input, neighbors_from_orders, INF, elapsed_time


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

    nodes = set()
    for e in element_bodies:
        for n in e:
            if element_supports(e, n, node_points):
                init.append(('Supports', e, n))
            if is_start_node(n, e, node_points):
                init.append(('StartNode', n, e))
        #if e[0] not in nodes:
        #    add_text(e[0], position=(0, 0, -0.02), parent=element_bodies[e])
        #if e[1] not in nodes:
        #    add_text(e[1], position=(0, 0, 0.02), parent=element_bodies[e])
        #nodes.update(e)

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
                  debug=False, max_time=30):
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
    #planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    planner = 'ff-eager-tiebreak' # Need to use a eager search, otherwise doesn't incorporate new cost
    #planner = 'max-astar'
    # TODO: limit the branching factor if necessary
    solution = solve_focused(pddlstream_problem, stream_info=stream_info, max_time=max_time,
                             effort_weight=1, unit_efforts=True, max_skeletons=None, unit_costs=True, bind=False,
                             planner=planner, max_planner_time=15, debug=debug, reorder=False)
    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    print_solution(solution)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(25)
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
    start_time = time.time()
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
                                 restarts=50, iterations=100, smooth=100)
        if path is None:
            print('Failed to find a motion plan!')
            return None
        motion_traj = MotionTrajectory(robot, movable_joints, path)
        print('{}) {} | Time: {:.3f}'.format(i, motion_traj, elapsed_time(start_time)))
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
    print('Trajectories:', len(trajectories))
    for i, trajectory in enumerate(trajectories):
        if isinstance(trajectory, PrintTrajectory):
            print(i, trajectory, trajectory.n1 in connected, trajectory.n2 in connected,
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
    # simple_frame | Nodes: 12 | Ground: 4 | Elements: 19
    # topopt-100 | Nodes: 88 | Ground: 20 | Elements: 132
    # topopt-205 | Nodes: 89 | Ground: 19 | Elements: 164
    # mars-bubble | Nodes: 97 | Ground: 11 | Elements: 225
    # djmm_test_block | Nodes: 76 | Ground: 13 | Elements: 253
    # voronoi | Nodes: 162 | Ground: 14 | Elements: 306
    # topopt-310 | Nodes: 160 | Ground: 39 | Elements: 310
    # sig_artopt-bunny | Nodes: 219 | Ground: 14 | Elements: 418
    parser.add_argument('-p', '--problem', default='simple_frame', help='The name of the problem to solve')
    parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions with obstacles')
    parser.add_argument('-m', '--motions', action='store_true', help='Plans motions between each extrusion')
    parser.add_argument('-t', '--max_time', default=INF, type=int, help='The max time')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    args = parser.parse_args()
    print('Arguments:', args)

    # TODO: setCollisionFilterGroupMask
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)

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

    with LockRenderer(False):
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

    # TODO: slow down automatically near endpoints
    # TODO: heuristic that orders elements by angle
    # TODO: check that both teh start and end satisfy
    # TODO: return to start when done


if __name__ == '__main__':
    main()

# TODO: only consider axioms that could be relevant

"""
306) print 161 (52, 161) 161->52
         267544651 function calls (265403829 primitive calls) in 952.845 seconds

   Ordered by: internal time
   List reduced from 1097 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
 12764657  537.943    0.000  537.943    0.000 {pybullet.getClosestPoints}
    74563  145.749    0.002  145.749    0.002 {method 'read' of 'file' objects}
   281510   13.870    0.000   13.870    0.000 {method 'items' of 'dict' objects}
 12056767   13.726    0.000  542.717    0.000 pddlstream/examples/pybullet/utils/pybullet_tools/utils.py:1894(pairwise_collision)
     1197   11.320    0.009  503.233    0.420 extrusion/extrusion_utils.py:32(check_trajectory_collision)
 18779836    8.199    0.000    8.199    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:238(key)
1297281/807655    7.912    0.000   22.393    0.000 {sorted}
       12    7.891    0.658   11.182    0.932 pddlstream/pddlstream/algorithms/scheduling/negative.py:59(recover_negative_axioms)
  9389918    7.624    0.000   15.823    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:241(__lt__)
       30    7.059    0.235   19.829    0.661 pddlstream/pddlstream/algorithms/scheduling/recover_streams.py:19(get_achieving_streams)
 10546685    6.035    0.000    6.035    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:230(__eq__)
  5512767    5.909    0.000    6.580    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:226(__init__)
    50456    4.312    0.000   13.445    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/axiom_rules.py:132(simplify)
  5002281    4.136    0.000    4.136    0.000 {built-in method __new__ of type object at 0x10b534bf8}
       17    3.788    0.223    4.280    0.252 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/variable_order.py:78(calculate_topological_pseudo_sort)
   766368    3.655    0.000    6.093    0.000 {_heapq.heappop}
 16614771    3.502    0.000    3.502    0.000 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/pddl/conditions.py:14(__hash__)
       18    3.171    0.176  348.696   19.372 pddlstream/pddlstream/algorithms/scheduling/plan_streams.py:91(plan_streams)
2592310/2175389    2.994    0.000   10.914    0.000 {map}
       34    2.965    0.087    4.285    0.126 /Users/caelan/Programs/LIS/git/collaborations/pb-construction/pddlstream/pddlstream/algorithms/../../FastDownward/builds/release64/bin/translate/translate.py:58(strips_to_sas_dictionary)
     3702    2.837    0.001    9.238    0.002 pddlstream/pddlstream/algorithms/instantiate_task.py:72(get_achieving_axioms)
       18    2.679    0.149   35.207    1.956 pddlstream/pddlstream/algorithms/instantiate_task.py:112(instantiate_domain)
  1368240    2.628    0.000    2.628    0.000 {zip}
  1080621    2.552    0.000    8.352    0.000 pddlstream/pddlstream/algorithms/downward.py:203(fd_from_evaluation)
  9856293    2.447    0.000    2.447    0.000 pddlstream/pddlstream/utils.py:207(__lt__)
"""
