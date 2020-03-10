from __future__ import print_function

from collections import defaultdict
from itertools import product

import numpy as np

from extrusion.heuristics import compute_layer_from_vertex, compute_layer_from_element
from extrusion.stream import get_print_gen_fn, USE_CONMECH
from extrusion.utils import load_robot, get_other_node, get_node_neighbors, PrintTrajectory, get_midpoint, \
    get_element_length, TOOL_VELOCITY, recover_sequence, flatten_commands
from extrusion.visualization import draw_ordered, label_nodes
from examples.pybullet.turtlebots.run import get_test_cfree_traj_traj
from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.algorithms.focused import solve_focused #, CURRENT_STREAM_PLAN
from pddlstream.language.constants import And, PDDLProblem, print_solution, DurativeAction, Equal
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.language.function import FunctionInfo
from pddlstream.utils import read, get_file_path, inclusive_range
from pddlstream.language.temporal import compute_duration, get_end
from pybullet_tools.utils import get_configuration, set_pose, Pose, Euler, Point, get_point, \
    get_movable_joints, set_joint_position, has_gui, WorldSaver, wait_if_gui, add_line, RED, \
    wait_for_duration, get_length, INF, step_simulation, LockRenderer, randomize, pairwise_collision, set_configuration

STRIPSTREAM_ALGORITHM = 'stripstream'
ROBOT_TEMPLATE = 'r{}'

# ----- Small -----
# extreme_beam_test
# extrusion_exp_L75.0
# four-frame
# simple_frame
# topopt-205_long_beam_test
# long_beam_test

# ----- Medium -----
# robarch_tree_S, robarch_tree_M
# topopt-101_tiny
# semi_sphere
# compas_fea_beam_tree_simp, compas_fea_beam_tree_S_simp, compas_fea_beam_tree_M_simp

##################################################

def simulate_printing(node_points, trajectories, time_step=0.1, speed_up=10.):
    print_trajectories = [traj for traj in trajectories if isinstance(traj, PrintTrajectory)]
    handles = []
    current_time = 0.
    current_traj = print_trajectories.pop(0)
    current_curve = current_traj.interpolate_tool(node_points, start_time=current_time)
    current_position = current_curve.y[0]
    while True:
        print('Time: {:.3f} | Remaining: {} | Segments: {}'.format(
            current_time, len(print_trajectories), len(handles)))
        end_time = current_curve.x[-1]
        if end_time < current_time + time_step:
            handles.append(add_line(current_position, current_curve.y[-1], color=RED))
            if not print_trajectories:
                break
            current_traj = print_trajectories.pop(0)
            current_curve = current_traj.interpolate_tool(node_points, start_time=end_time)
            current_position = current_curve.y[0]
            print('New trajectory | Start time: {:.3f} | End time: {:.3f} | Duration: {:.3f}'.format(
                current_curve.x[0], current_curve.x[-1], current_curve.x[-1] - current_curve.x[0]))
        else:
            current_time += time_step
            new_position = current_curve(current_time)
            handles.append(add_line(current_position, new_position, color=RED))
            current_position = new_position
            wait_for_duration(time_step / speed_up)
            # wait_if_gui()
    wait_if_gui()
    return handles

def simulate_parallel(plan, time_step=0.1, speed_up=10.):
    # TODO: ensure the step size is appropriate
    makespan = compute_duration(plan)
    print('\nMakespan: {:.3f}'.format(makespan))
    trajectories = []
    for action in plan:
        command = action.args[-1]
        # Regression planning
        start_time = makespan - get_end(action)
        #end_time = makespan - action.start
        command.retime(start_time=start_time)
        #print(action)
        #print(start_time, end_time, action.duration)
        #print(command.start_time, command.end_time, command.duration)
        #for traj in command.trajectories:
        #    print(traj, traj.start_time, traj.end_time, traj.duration)
        trajectories.extend(command.trajectories)
    #print(sum(traj.duration for traj in trajectories))

    wait_if_gui('Begin?')
    for t in inclusive_range(0, makespan, time_step):
        print('t={:.3f}/{:.3f}'.format(t, makespan))
        # if action.start <= t <= get_end(action):
        for traj in trajectories:
            conf = traj.at(t)
            #if conf is not None:
            #    print(traj, conf)
        #step_simulation()
        wait_for_duration(time_step / speed_up)
    wait_if_gui('Finish?')

##################################################

def compute_directions(elements, layer_from_n):
    directions = set()
    for e in elements:
        for n1 in e:
            n2 = get_other_node(n1, e)
            if layer_from_n[n1] <= layer_from_n[n2]:
                directions.add((n1, e, n2))
    return directions

def compute_local_orders(elements, layer_from_n):
    # TODO: could make level objects
    # Could update whether a node is connected, but it's slightly tricky
    partial_orders = set()
    for n1, neighbors in get_node_neighbors(elements).items():
        below, equal, above = [], [], []  # wrt n1
        for e in neighbors:  # Directed version of this (likely wouldn't need directions then)
            n2 = get_other_node(n1, e)
            if layer_from_n[n1] < layer_from_n[n2]:
                above.append(e)
            elif layer_from_n[n1] > layer_from_n[n2]:
                below.append(e)
            else:
                equal.append(e)
        partial_orders.update(product(below, equal + above))
        partial_orders.update(product(equal, above))
    return partial_orders

def compute_global_orders(element_bodies, node_points, ground_nodes):
    # TODO: further bucket
    # TODO: separate orders per robot
    layer_from_e = compute_layer_from_element(element_bodies, node_points, ground_nodes)
    elements_from_layer = defaultdict(list)
    for e, l in layer_from_e.items():
        elements_from_layer[l].append(e)
    partial_orders = set()
    layers = sorted(elements_from_layer)
    for layer in layers[:-1]:
        partial_orders.update(product(elements_from_layer[layer], elements_from_layer[layer+1]))
    return partial_orders

##################################################

def get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes,
                   trajectories=[], temporal=True, local=False, **kwargs):
    # TODO: TFD submodule
    elements = set(element_bodies)
    layer_from_n = compute_layer_from_vertex(element_bodies, node_points, ground_nodes)

    directions = compute_directions(elements, layer_from_n)
    #partial_orders = set()
    if local:
        # makespan seems more effective than CEA
        partial_orders = compute_local_orders(elements, layer_from_n) # makes the makespan heuristic slow
    else:
        partial_orders = compute_global_orders(element_bodies, node_points, ground_nodes)

    #print(supports)
    # draw_model(supporters, node_points, ground_nodes, color=RED)
    # wait_if_gui()

    initial_confs = {ROBOT_TEMPLATE.format(i): np.array(get_configuration(robot)) for i, robot in enumerate(robots)}

    domain_pddl = read(get_file_path(__file__, 'pddl/temporal.pddl' if temporal else 'pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    constant_map = {}

    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-print': get_wild_print_gen_fn(robots, obstacles, node_points, element_bodies, ground_nodes,
                                              partial_orders=partial_orders, **kwargs),
        #'test-stiffness': from_test(test_stiffness),
        #'test-cfree-traj-conf': from_test(lambda *args: True),
        #'test-cfree-traj-traj': from_test(get_cfree_test(**kwargs)),

        'TrajTrajCollision': get_collision_test(robots, *kwargs),
        #'Length': lambda e: get_element_length(e, node_points),
        'Distance': lambda r, t: t.get_link_distance(),
    }

    init = [
        Equal(('Speed',), TOOL_VELOCITY),
    ]
    for robot, conf in initial_confs.items():
        init.extend([
            ('Robot', robot),
            ('Conf', robot, conf),
            ('AtConf', robot, conf),
            ('Idle', robot),
            #('CanMove', robot),
        ])

    assignments = set()
    for element in elements:
        point = get_midpoint(node_points, element)
        closest_robot, closest_distance = None, INF
        for i, robot in enumerate(robots):
            base_point = get_point(robot)
            distance = get_length((base_point - point)[:2])
            if distance < closest_distance:
                closest_robot, closest_distance = ROBOT_TEMPLATE.format(i), distance
        assert closest_robot is not None
        # TODO: assign to several robots if close to the best distance
        assignments.add((closest_robot, element))

    init.extend(('Grounded', n) for n in ground_nodes)
    init.extend(('Direction',) + triplet for triplet in directions)
    init.extend(('Order',) + pair for pair in partial_orders)
    init.extend(('Assigned',) + pair for pair in assignments)
    # TODO: only move actions between adjacent layers

    for e in elements:
        n1, n2 = e
        #n1, n2 = ['n{}'.format(i) for i in e]
        init.extend([
            ('Node', n1),
            ('Node', n2),
            ('Element', e),
            ('Printed', e),
        ])

    assert not trajectories
    # for t in trajectories:
    #     init.extend([
    #         ('Traj', t),
    #         ('PrintAction', t.n1, t.element, t),
    #     ])

    goal_literals = [
        #('AtConf', robot, initial_conf),
    ]
    #goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Removed', e) for e in element_bodies)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

def mirror_robot(robot1, node_points):
    centroid = np.average(node_points, axis=0)
    # print(centroid)
    # print(get_point(robot1))
    robot2 = load_robot()
    set_pose(robot2, Pose(point=Point(*2 * centroid[:2]), euler=Euler(yaw=np.pi)))

    # robots = [robot1]
    robots = [robot1, robot2]
    for robot in robots:
        joint1 = get_movable_joints(robot)[0]
        set_joint_position(robot, joint1, np.pi / 8)
    return robots

def plan_sequence(robot1, obstacles, node_points, element_bodies, ground_nodes,
                  trajectories=[], collisions=True, disable=False, max_time=30, checker=None):
    if trajectories is None:
        return None
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    # TODO: only consider axioms that could be relevant
    # TODO: iterated search using random restarts
    # TODO: most of the time seems to be spent extracting the stream plan
    # TODO: NEGATIVE_SUFFIX to make axioms easier
    # TODO: sort by action cost heuristic
    # http://www.fast-downward.org/Doc/Evaluator#Max_evaluator

    robots = mirror_robot(robot1, node_points)
    saver = WorldSaver()
    pddlstream_problem = get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes,
                                        trajectories=trajectories, collisions=collisions, disable=disable,
                                        precompute_collisions=True)
    print('Init:', pddlstream_problem.init)
    print('Goal:', pddlstream_problem.goal)

    min_length = min(get_element_length(e, node_points) for e in element_bodies)
    max_length = max(get_element_length(e, node_points) for e in element_bodies)
    print('Min length: {} | Max length: {}'.format(min_length, max_length))
    #opt_distance = min_length # Admissible
    opt_distance = max_length # Inadmissible/greedy

    stream_info = {
        'sample-print': StreamInfo(PartialInputs(unique=True)),
        'test-cfree-traj-conf': StreamInfo(p_success=1e-2, negate=True), #, verbose=False),
        'test-cfree-traj-traj': StreamInfo(p_success=1e-2, negate=True),
        'TrajTrajCollision': FunctionInfo(p_success=1e-1, overhead=1), # TODO: verbose
        'Length': FunctionInfo(eager=True),  # Need to eagerly evaluate otherwise 0 makespan (failure)
        'Distance': FunctionInfo(opt_fn=lambda r, t: opt_distance, eager=True), # TODO: use the corresponding element length
    }

    # TODO: goal serialization
    # TODO: could revert back to goal count now that no deadends
    # TODO: limit the branching factor if necessary
    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    set_cost_scale(1)
    #planner = 'ff-ehc'
    #planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    planner = 'ff-eager-tiebreak' # Need to use a eager search, otherwise doesn't incorporate child cost
    #planner = 'max-astar'

    with LockRenderer(lock=False):
        #solution = solve_incremental(pddlstream_problem, planner='add-random-lazy', max_time=600,
        #                             max_planner_time=300, debug=True)
        solution = solve_focused(pddlstream_problem, stream_info=stream_info, max_time=max_time,
                                 effort_weight=None, unit_efforts=True, max_skeletons=None, unit_costs=False, bind=False,
                                 planner=planner, max_planner_time=60, debug=True, reorder=False,
                                 initial_complexity=1)

    print_solution(solution)
    plan, _, _ = solution
    data = {}
    if plan is None:
        return None, data

    if plan and not isinstance(plan[0], DurativeAction):
        time_from_start = 0.
        retimed_plan = []
        for name, args in plan:
            command = args[-1]
            command.retime(start_time=time_from_start)
            duration = command.duration
            retimed_plan.append(DurativeAction(name, args, time_from_start, duration))
            time_from_start += duration
        plan = retimed_plan

    if has_gui():
        saver.restore()
        # label_nodes(node_points)
        # commands = [action.args[-1] for action in reversed(plan) if action.name == 'print']
        # trajectories = flatten_commands(commands)
        # elements = recover_sequence(trajectories)
        # draw_ordered(elements, node_points)
        # wait_if_gui('Continue?')

        #simulate_printing(node_points, trajectories)
        #display_trajectories(node_points, ground_nodes, trajectories)
        simulate_parallel(plan)

    return None, data
    #return trajectories, data

##################################################

def index_from_name(robots, name):
    return robots[int(name[1:])]

def get_wild_print_gen_fn(robots, obstacles, node_points, element_bodies, ground_nodes,
                          collisions=True, **kwargs):
    # TODO: could reuse end-effector trajectories
    gen_fn_from_robot = {robot: get_print_gen_fn(robot, obstacles, node_points, element_bodies,
                                                 ground_nodes, **kwargs) for robot in robots}
    def wild_gen_fn(name, node1, element, node2):
        # TODO: could cache this
        # sequence = [result.get_mapping()['?e'].value for result in CURRENT_STREAM_PLAN]
        # index = sequence.index(element)
        # printed = sequence[:index]
        # TODO: this might need to be recomputed per iteration
        # TODO: condition on plan/downstream constraints
        # TODO: stream fusion
        robot = index_from_name(robots, name)
        #generator = gen_fn_from_robot[robot](node1, element)
        for command, in gen_fn_from_robot[robot](node1, element):
            q1 = np.array(command.start_conf)
            q2 = np.array(command.end_conf)
            outputs = [(q1, q2, command)]
            facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
            yield WildOutput(outputs, facts)
    return wild_gen_fn

def get_collision_test(robots, collisions=True, *args, **kwargs):
    # TODO: remove *args
    def test(name1, traj1, name2, traj2):
        robot1 = index_from_name(robots, name1)
        robot2 = index_from_name(robots, name2)
        if (robot1 == robot2) or not collisions:
            return False
        for conf1 in randomize(traj1.trajectories[1].path):
            #if not aabb_overlap(get_turtle_traj_aabb(conf1), get_turtle_traj_aabb(traj2)):
            #    continue
            set_configuration(robot1, conf1)
            for conf2 in randomize(traj2.trajectories[1].path):
                #if not aabb_overlap(get_turtle_traj_aabb(conf1), get_turtle_traj_aabb(conf2)):
                #    continue
                set_configuration(robot2, conf2)
                if pairwise_collision(robot1, robot2):
                    return True
        return False
    return test

def test_stiffness(fluents=[]):
    assert all(fact[0] == 'printed' for fact in fluents)
    if not USE_CONMECH:
       return True
    # https://github.com/yijiangh/conmech
    # TODO: to use the non-skeleton focused algorithm, need to remove the negative axiom upon success
    elements = {fact[1] for fact in fluents}
    #print(elements)
    return True