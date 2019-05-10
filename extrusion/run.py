#!/usr/bin/env python
from __future__ import print_function

import sys

sys.path.append('pddlstream/')

import argparse

from extrusion.motion import compute_motions, display_trajectories
from extrusion.sorted import heuristic_planner
from extrusion.stripstream import plan_sequence
from extrusion.utils import load_world, create_stiffness_checker, \
    downsample_nodes, check_connected, get_connected_structures, check_stiffness
from extrusion.parsing import load_extrusion, draw_element, create_elements, get_extrusion_path
from extrusion.stream import get_print_gen_fn
from extrusion.greedy import regression, progression

from examples.pybullet.utils.pybullet_tools.utils import connect, disconnect, get_movable_joints, add_text, \
    get_joint_positions, LockRenderer, wait_for_user, has_gui

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
    checker = create_stiffness_checker(extrusion_name)

    # TODO: construct the structure in different ways (random, connected)
    handles = []
    all_connected = True
    all_stiff = True
    extruded_elements = set()
    for element in planned_elements:
        extruded_elements.add(element)
        is_connected = check_connected(ground_nodes, extruded_elements)
        structures = get_connected_structures(extruded_elements)
        is_stiff = check_stiffness(checker, element_from_id, extruded_elements)
        all_stiff &= is_stiff
        print('Elements: {} | Structures: {} | Connected: {} | Stiff: {}'.format(
            len(extruded_elements), len(structures), is_connected, is_stiff))
        is_stable = is_connected and is_stiff
        if has_gui():
            color = (0, 1, 0) if is_stable else (1, 0, 0)
            handles.append(draw_element(node_points, element, color))
            if not is_stable:
                wait_for_user()
    return all_connected and all_stiff


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
    # djmm_bridge | Nodes: 1548 | Ground: 258 | Elements: 6427
    # djmm_test_block | Nodes: 76 | Ground: 13 | Elements: 253
    parser.add_argument('-a', '--algorithm', default='stripstream', help='Which algorithm to use')
    parser.add_argument('-p', '--problem', default='simple_frame', help='The name of the problem to solve')
    parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions with obstacles')
    parser.add_argument('-m', '--motions', action='store_true', help='Plans motions between each extrusion')
    parser.add_argument('-d', '--disable', action='store_true', help='Disables trajectory planning')
    parser.add_argument('-t', '--max_time', default=INF, type=int, help='The max time')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    args = parser.parse_args()
    print('Arguments:', args)

    # TODO: setCollisionFilterGroupMask
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    element_from_id, node_points, ground_nodes = load_extrusion(args.problem)
    elements = list(element_from_id.values())
    elements, ground_nodes = downsample_nodes(elements, node_points, ground_nodes)

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
        if args.algorithm == 'stripstream':
            planned_trajectories = plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                                                 trajectories=trajectories, collisions=not args.cfree,
                                                 disable=args.disable, max_time=args.max_time)
        elif args.algorithm == 'regression':
            planned_trajectories = regression(robot, obstacles, element_bodies, args.problem, disable=args.disable)
        elif args.algorithm == 'progression':
            planned_trajectories = progression(robot, obstacles, element_bodies, args.problem, disable=args.disable)
        elif args.algorithm == 'heuristic':
            planned_trajectories = heuristic_planner(robot, obstacles, node_points, element_bodies,
                                                     ground_nodes, disable=args.disable)
        else:
            raise ValueError(args.algorithm)
        planned_elements = [traj.element for traj in planned_trajectories]
        if args.motions:
            planned_trajectories = compute_motions(robot, obstacles, element_bodies, initial_conf, planned_trajectories)
    disconnect()

    #random.shuffle(planned_elements)
    #planned_elements = sorted(elements, key=lambda e: max(node_points[n][2] for n in e)) # TODO: tiebreak by angle or x

    connect(use_gui=True)
    floor, robot = load_world()
    print(check_plan(args.problem, planned_elements))
    if args.disable:
        wait_for_user()
        return
    disconnect()
    display_trajectories(ground_nodes, planned_trajectories)
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
