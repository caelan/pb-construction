from __future__ import print_function

from collections import defaultdict, Counter
from itertools import product, permutations
from operator import itemgetter

import numpy as np
import time
import os

from extrusion.heuristics import compute_layer_from_vertex, compute_distance_from_node
from extrusion.stream import get_print_gen_fn, USE_CONMECH, APPROACH_DISTANCE, SELF_COLLISIONS, \
    JOINT_WEIGHTS, RESOLUTION
from extrusion.utils import load_robot, get_other_node, get_node_neighbors, PrintTrajectory, get_midpoint, \
    get_element_length, TOOL_VELOCITY, Command, MotionTrajectory, \
    get_disabled_collisions, retrace_supporters
from extrusion.visualization import set_extrusion_camera, draw_model
#from examples.pybullet.turtlebots.run import *
from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.focused import solve_focused #, CURRENT_STREAM_PLAN
from pddlstream.language.constants import And, PDDLProblem, print_solution, DurativeAction, Equal, print_plan, \
    NOT, EQ, get_prefix, get_function
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.language.function import FunctionInfo
from pddlstream.utils import read, get_file_path, inclusive_range, neighbors_from_orders
from pddlstream.language.temporal import compute_duration, compute_start, compute_end, apply_start, \
    create_planner, DURATIVE_ACTIONS, reverse_plan, get_tfd_path
from pybullet_tools.utils import get_configuration, set_pose, Euler, get_point, \
    get_movable_joints, has_gui, WorldSaver, wait_if_gui, add_line, BLUE, RED, \
    wait_for_duration, get_length, INF, LockRenderer, randomize, set_configuration, Pose, Point, aabb_overlap, pairwise_link_collision, \
    aabb_union, plan_joint_motion, SEPARATOR, user_input, remove_all_debug, GREEN, elapsed_time, VideoSaver, \
    point_from_pose, draw_aabb, get_pose, tform_point, invert, get_yaw, draw_pose

STRIPSTREAM_ALGORITHM = 'stripstream'
ROBOT_TEMPLATE = 'r{}'

DUAL_CONF = [np.pi/4, -np.pi/4, np.pi/2, 0, np.pi/4, -np.pi/2] # np.pi/8

CUSTOM_LIMITS = { # TODO: do instead of modifying the URDF
   #'robot_joint_a1': (-np.pi/2, np.pi/2),
   #'robot_joint_a1': (-np.pi / 2, 0),
}

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

def index_from_name(robots, name):
    return robots[int(name[1:])]

#def name_from_index(i):
#    return ROBOT_TEMPLATE.format(i)

class Conf(object):
    def __init__(self, robot, positions=None, node=None, element=None):
        self.robot = robot
        self.joints = get_movable_joints(self.robot)
        if positions is None:
            positions = get_configuration(self.robot)
        self.positions = positions
        self.node = node
        self.element = element
    def assign(self):
        set_configuration(self.robot, self.positions)
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.node)

def mirror_robot(robot1, node_points):
    # TODO: place robots side by side or diagonal across
    set_extrusion_camera(node_points, theta=-np.pi/3)
    #draw_pose(Pose())
    centroid = np.average(node_points, axis=0)
    centroid_pose = Pose(point=centroid)
    #draw_pose(Pose(point=centroid))

    # print(centroid)
    scale = 0. # 0.15
    vector = get_point(robot1) - centroid
    set_pose(robot1, Pose(point=Point(*+scale*vector[:2])))
    # Inner product of end-effector z with base->centroid or perpendicular to this line
    # Partition by sides

    robot2 = load_robot()
    set_pose(robot2, Pose(point=Point(*-(2+scale)*vector[:2]), euler=Euler(yaw=np.pi)))

    # robots = [robot1]
    robots = [robot1, robot2]
    for robot in robots:
        set_configuration(robot, DUAL_CONF)
        # joint1 = get_movable_joints(robot)[0]
        # set_joint_position(robot, joint1, np.pi / 8)
        draw_pose(get_pose(robot), length=0.25)
    return robots

##################################################

def simulate_printing(node_points, trajectories, time_step=0.1, speed_up=10.):
    # TODO: deprecate
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
            # TODO: longer wait for recording videos
            wait_for_duration(time_step / speed_up)
            # wait_if_gui()
    wait_if_gui()
    return handles

##################################################

def simulate_parallel(robots, plan, time_step=0.1, speed_up=10., record=None): # None | video.mp4
    # TODO: ensure the step size is appropriate
    makespan = compute_duration(plan)
    print('\nMakespan: {:.3f}'.format(makespan))
    if plan is None:
        return
    trajectories = []
    for action in plan:
        command = action.args[-1]
        if (action.name == 'move') and (command.start_conf is action.args[-2].positions):
            command = command.reverse()
        command.retime(start_time=action.start)
        #print(action)
        #print(action.start, get_end(action), action.duration)
        #print(command.start_time, command.end_time, command.duration)
        #for traj in command.trajectories:
        #    print(traj, traj.start_time, traj.end_time, traj.duration)
        trajectories.extend(command.trajectories)
    #print(sum(traj.duration for traj in trajectories))
    num_motion = sum(action.name == 'move' for action in plan)

    wait_if_gui('Begin?')
    with VideoSaver(record):
        for t in inclusive_range(0, makespan, time_step):
            # if action.start <= t <= get_end(action):
            executing = Counter(traj.robot  for traj in trajectories if traj.at(t) is not None)
            print('t={:.3f}/{:.3f} | executing={}'.format(t, makespan, executing))
            for robot in robots:
                num = executing.get(robot, 0)
                if 2 <= num:
                    raise RuntimeError('Robot {} simultaneously executing {} trajectories'.format(robot, num))
                if (num_motion == 0) and (num == 0):
                    set_configuration(robot, DUAL_CONF)
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

def compute_elements_from_layer(elements, layer_from_n):
    #layer_from_e = compute_layer_from_element(element_bodies, node_points, ground_nodes)
    layer_from_e = {e: min(layer_from_n[v] for v in e) for e in elements}
    elements_from_layer = defaultdict(set)
    for e, l in layer_from_e.items():
        elements_from_layer[l].add(e)
    return elements_from_layer

def compute_global_orders(elements, layer_from_n):
    # TODO: separate orders per robot
    elements_from_layer = compute_elements_from_layer(elements, layer_from_n)
    partial_orders = set()
    layers = sorted(elements_from_layer)
    for layer in layers[:-1]:
        partial_orders.update(product(elements_from_layer[layer], elements_from_layer[layer+1]))
    return partial_orders

##################################################

def cluster_vertices(elements, node_points, ground_nodes, ratio=0.25, weight=0.):
    # TODO: incorporate x,y,z proximity with a lower weight as well
    from sklearn.cluster import KMeans
    #nodes = nodes_from_elements(elements)
    node_from_vertex = compute_distance_from_node(elements, node_points, ground_nodes)
    nodes = sorted(node_from_vertex)
    costs = [node_from_vertex[node].cost for node in nodes]
    # TODO: use element midpoints

    num = int(np.ceil(ratio*len(nodes)))
    model = KMeans(n_clusters=num, n_init=10, max_iter=300, tol=1e-4)
    xx = [[cost] for cost in costs]
    pp = model.fit_predict(xx)

    frequencies = Counter(pp)
    print('# nodes: {} | # elements: {} | max clusters: {} | # clusters: {}'.format(
        len(nodes), len(elements), num, len(frequencies)))
    #print(frequencies)
    #print(sorted(costs))
    #print(sorted(model.cluster_centers_))

    clusters = sorted(range(len(model.cluster_centers_)), key=lambda c: model.cluster_centers_[c][0])
    index_from_cluster = dict(zip(clusters, range(len(clusters))))

    cluster_from_node = {node: index_from_cluster[cluster] for node, cluster in zip(nodes, pp)}
    elements_from_clusters = {}
    for element in elements:
        cluster = min(map(cluster_from_node.get, element))
        elements_from_clusters.setdefault(cluster, set()).add(element)
    #directions = compute_directions(elements, cluster_from_node)

    #colors = sample_colors(len(elements_from_clusters))
    #for cluster, color in zip(sorted(elements_from_clusters), colors):
    #    draw_model(elements_from_clusters[cluster], node_points, ground_nodes, color=color)
    #wait_if_gui()
    return cluster_from_node

def compute_assignments(robots, elements, node_points, initial_confs):
    # TODO: print direction might influence the assignment
    assignments = {name: set() for name in initial_confs}
    for element in elements:
        point = get_midpoint(node_points, element) # min/max
        closest_robot, closest_distance = None, INF
        for i, robot in enumerate(robots):
            base_pose = get_pose(robot)
            base_point = point_from_pose(base_pose)
            point_base = tform_point(invert(base_pose), point)
            distance = get_yaw(point_base) # which side its on
            #distance = abs((base_point - point)[0]) # x distance
            #distance = get_length((base_point - point)[:2]) # xy distance
            if distance < closest_distance:
                closest_robot, closest_distance = ROBOT_TEMPLATE.format(i), distance
        assert closest_robot is not None
        # TODO: assign to several robots if close to the best distance
        assignments[closest_robot].add(element)
    return assignments

def compute_transits(layer_from_n, directions):
    # TODO: remove any extrusion pairs
    # TODO: use the partial orders instead
    transits = []
    for (n0, e1, n1), (n2, e2, _) in permutations(directions, r=2):
        # TODO: an individual robot technically could jump two levels
        if layer_from_n[n2] - layer_from_n[n0] in [0, 1]: # TODO: robot centric?
            transits.append((e1, n1, n2, e2))
    return transits

##################################################

def get_opt_distance_fn(element_bodies, node_points):
    min_length = min(get_element_length(e, node_points) for e in element_bodies)
    max_length = max(get_element_length(e, node_points) for e in element_bodies)
    print('Min length: {} | Max length: {}'.format(min_length, max_length))
    # opt_distance = min_length # Admissible
    opt_distance = max_length + 2 * APPROACH_DISTANCE  # Inadmissible/greedy

    def fn(robot, command):
        # TODO: use the corresponding element length
        if command.stream == 'sample-move':
            #e1, n1, n2, e2 = command.input_objects[-4:]
            r, q1, q2 = command.input_objects[:3] # TODO: straight-line distance
            return 2.
        elif command.stream == 'sample-print':
            return opt_distance
        else:
            raise NotImplementedError(command.stream)
    return fn

def get_pddlstream(robots, static_obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                   initial_confs={}, printed=set(), removed=set(),
                   additional_init=[], additional_orders=set(), trajectories=[],
                   temporal=True, sequential=False, local=False,
                   can_print=True, can_transit=False,
                   checker=None, **kwargs):
    try:
        get_tfd_path()
    except RuntimeError:
        temporal = False
        print('Temporal Fast Downward is not installed. Disabling temporal planning.')
    # TODO: TFD submodule
    assert not removed & printed
    remaining = set(element_bodies) - removed - printed
    element_obstacles = {element_bodies[e] for e in printed}
    obstacles = set(static_obstacles) | element_obstacles
    max_layer = max(layer_from_n.values())

    directions = compute_directions(remaining, layer_from_n)
    if local:
        # makespan seems more effective than CEA
        partial_orders = compute_local_orders(remaining, layer_from_n) # makes the makespan heuristic slow
    else:
        partial_orders = compute_global_orders(remaining, layer_from_n)
    partial_orders.update(additional_orders)
    # draw_model(supporters, node_points, ground_nodes, color=RED)
    # wait_if_gui()

    domain_pddl = read(get_file_path(__file__, os.path.join('pddl', 'temporal.pddl' if temporal else 'domain.pddl')))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    constant_map = {}

    # TODO: don't evaluate TrajTrajCollision until the plan is retimed
    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-move': get_wild_move_gen_fn(robots, obstacles, element_bodies,
                                            partial_orders=partial_orders, **kwargs),
        'sample-print': get_wild_print_gen_fn(robots, obstacles, node_points, element_bodies, ground_nodes,
                                              initial_confs=initial_confs, partial_orders=partial_orders,
                                              removed=removed, **kwargs),
        #'test-stiffness': from_test(test_stiffness),
        #'test-cfree-traj-conf': from_test(lambda *args: True),
        #'test-cfree-traj-traj': from_test(get_cfree_test(**kwargs)),

        'TrajTrajCollision': get_collision_test(robots, **kwargs),
        'TrajConfCollision': lambda *args, **kwargs: False, # TODO: could treat conf as length 1 traj

        'Length': lambda e: get_element_length(e, node_points),
        'Distance': lambda r, t: t.get_link_distance(),
        'Duration': lambda r, t: t.get_link_distance() / TOOL_VELOCITY,
        'Euclidean': lambda n1, n2: get_element_length((n1, n2), node_points),
    }

    assignments = compute_assignments(robots, remaining, node_points, initial_confs)
    transits = compute_transits(layer_from_n, directions)

    init = [
        ('Stationary',),
        Equal(('Speed',), TOOL_VELOCITY),
    ]
    init.extend(additional_init)
    if can_print:
        init.append(('Print',))
    if can_transit:
        init.append(('Move',))
    if sequential:
        init.append(('Sequential',))
    for name, conf in initial_confs.items():
        robot = index_from_name(robots, name)
        #init_node = -robot
        init_node = '{}-q0'.format(robot)
        init.extend([
            #('Node', init_node),
            ('BackoffConf', name, conf),
            ('Robot', name),
            ('Conf', name, conf),
            ('AtConf', name, conf),
            ('Idle', name),
            #('CanMove', name),
            #('Start', name, init_node, None, conf),
            #('End', name, None, init_node, conf),
        ])
        for (n1, e, n2) in directions:
            if layer_from_n[n1] == 0:
                transits.append((None, init_node, n1, e))
            if layer_from_n[n2] == max_layer:
                transits.append((e, n2, init_node, None))

    #init.extend(('Grounded', n) for n in ground_nodes)
    init.extend(('Direction',) + tup for tup in directions)
    init.extend(('Order',) + tup for tup in partial_orders)
    # TODO: can relax assigned if I go by layers
    init.extend(('Assigned', r, e) for r in assignments for e in assignments[r])
    #init.extend(('Transit',) + tup for tup in transits)
    # TODO: only move actions between adjacent layers

    for e in remaining:
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

    goal_literals = []
    if can_transit:
        goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Removed', e) for e in remaining)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

GREEDY_PLANNER = create_planner(greedy=False, lazy=True, h_cea=True)
POSTPROCESS_PLANNER = create_planner(anytime=True, greedy=False, lazy=True, h_cea=False, h_makespan=True)
#POSTPROCESS_PLANNER = create_planner(anytime=True, lazy=False, h_makespan=True)

def solve_pddlstream(problem, node_points, element_bodies, planner=GREEDY_PLANNER, max_time=60):
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    # TODO: only consider axioms that could be relevant
    # TODO: iterated search using random restarts
    # TODO: most of the time seems to be spent extracting the stream plan
    # TODO: NEGATIVE_SUFFIX to make axioms easier
    # TODO: sort by action cost heuristic
    # http://www.fast-downward.org/Doc/Evaluator#Max_evaluator

    temporal = DURATIVE_ACTIONS in problem.domain_pddl
    print('Init:', problem.init)
    print('Goal:', problem.goal)
    print('Max time:', max_time)
    print('Temporal:', temporal)

    stream_info = {
        # TODO: stream effort
        'sample-print': StreamInfo(PartialInputs(unique=True)),
        'sample-move': StreamInfo(PartialInputs(unique=True)),

        'test-cfree-traj-conf': StreamInfo(p_success=1e-2, negate=True),  # , verbose=False),
        'test-cfree-traj-traj': StreamInfo(p_success=1e-2, negate=True),
        'TrajConfCollision': FunctionInfo(p_success=1e-1, overhead=1),  # TODO: verbose
        'TrajTrajCollision': FunctionInfo(p_success=1e-1, overhead=1),  # TODO: verbose

        'Distance': FunctionInfo(opt_fn=get_opt_distance_fn(element_bodies, node_points), eager=True)
        # 'Length': FunctionInfo(eager=True),  # Need to eagerly evaluate otherwise 0 makespan (failure)
        # 'Duration': FunctionInfo(opt_fn=lambda r, t: opt_distance / TOOL_VELOCITY, eager=True),
        # 'Euclidean': FunctionInfo(eager=True),
    }

    # TODO: goal serialization
    # TODO: could revert back to goal count now that no deadends
    # TODO: limit the branching factor if necessary
    # TODO: ensure that function costs aren't prunning plans
    if not temporal:
        # Reachability heuristics good for detecting dead-ends
        # Infeasibility from the start means disconnected or collision
        set_cost_scale(1)
        # planner = 'ff-ehc'
        # planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
        planner = 'ff-eager-tiebreak'  # Need to use a eager search, otherwise doesn't incorporate child cost
        # planner = 'max-astar'

    # TODO: assert (instance.value == value)
    with LockRenderer(lock=False):
        # solution = solve_incremental(problem, planner='add-random-lazy', max_time=600,
        #                              max_planner_time=300, debug=True)
        # TODO: allow some types of failures
        solution = solve_focused(problem, stream_info=stream_info, max_time=max_time,
                                 effort_weight=None, unit_efforts=True, unit_costs=False, # TODO: effort_weight=None vs 0
                                 max_skeletons=None, bind=True, max_failures=INF,  # 0 | INF
                                 planner=planner, max_planner_time=60, debug=False, reorder=False,
                                 initial_complexity=1)

    print_solution(solution)
    plan, _, certificate = solution
    # TODO: post-process by calling planner again
    # TODO: could solve for trajectories conditioned on the sequence
    return plan, certificate

##################################################

def partition_plan(plan):
    plan_from_robot = defaultdict(list)
    for action in plan:
        plan_from_robot[action.args[0]].append(action)
    return plan_from_robot

def compute_total_orders(plan):
    # TODO: partially order trajectories
    partial_orders = set()
    for name, actions in partition_plan(plan).items():
        last_element = None
        for i, action in enumerate(actions):
            if action.name == 'print':
                r, n1, e, n2, q1, q2, t = action.args
                if last_element is not None:
                    # TODO: need level orders to synchronize between robots
                    # TODO: useful for collision checking
                    partial_orders.add((e, last_element))
                last_element = e
            else:
                raise NotImplementedError(action.name)
    return partial_orders

def extract_static_facts(plan, certificate, initial_confs):
    # TODO: use certificate instead
    # TODO: only keep objects used on the plan
    #static_facts = []
    static_facts = [f for f in certificate.all_facts if get_prefix(get_function(f))
                    in ['distance', 'trajtrajcollision']]
    for name, actions in partition_plan(plan).items():
        last_element = None
        last_conf = initial_confs[name]
        for i, action in enumerate(actions):
            if action.name == 'print':
                r, n1, e, n2, q1, q2, t = action.args
                static_facts.extend([
                    ('PrintAction',) + action.args,
                    ('Assigned', r, e),
                    ('Conf', r, q1),
                    ('Conf', r, q2),
                    ('Traj', r, t),
                    ('CTraj', r, t),
                    # (Start ?r ?n1 ?e ?q1) (End ?r ?e ?n2 ?q2)
                    ('Transition', r, q2, last_conf),
                ])
                if last_element is not None:
                    static_facts.append(('Order', e, last_element))
                last_element = e
                last_conf = q1
                # TODO: save collision information
            else:
                raise NotImplementedError(action.name)
        static_facts.extend([
            ('Transition', name, initial_confs[name], last_conf),
        ])
    return static_facts

##################################################

def solve_joint(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                trajectories=[], collisions=True, disable=False, max_time=INF, **kwargs):
    problem = get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                             trajectories=trajectories, collisions=collisions, disable=disable, **kwargs)
    return solve_pddlstream(problem, node_points, element_bodies, max_time=max_time)

def solve_serialized(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                     trajectories=[], post_process=False, collisions=True, disable=False, max_time=INF, **kwargs):
    start_time = time.time()
    saver = WorldSaver()
    elements = set(element_bodies)
    elements_from_layers = compute_elements_from_layer(elements, layer_from_n)
    layers = sorted(elements_from_layers.keys())
    print('Layers:', layers)

    full_plan = []
    makespan = 0.
    removed = set()
    for layer in reversed(layers):
        print(SEPARATOR)
        print('Layer: {}'.format(layer))
        saver.restore()
        remaining = elements_from_layers[layer]
        printed = elements - remaining - removed
        draw_model(remaining, node_points, ground_nodes, color=GREEN)
        draw_model(printed, node_points, ground_nodes, color=RED)
        problem = get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                                 printed=printed, removed=removed, return_home=False,
                                 trajectories=trajectories, collisions=collisions, disable=disable, **kwargs)
        layer_plan, certificate = solve_pddlstream(problem, node_points, element_bodies,
                                                   max_time=max_time - elapsed_time(start_time))
        remove_all_debug()
        if layer_plan is None:
            return None

        if post_process:
            print(SEPARATOR)
            # Allows the planner to continue to check collisions
            problem.init[:] = certificate.all_facts
            #static_facts = extract_static_facts(layer_plan, ...)
            #problem.init.extend(('Order',) + pair for pair in compute_total_orders(layer_plan))
            for fact in [('print',), ('move',)]:
                if fact in problem.init:
                    problem.init.remove(fact)
            new_layer_plan, _ = solve_pddlstream(problem, node_points, element_bodies,
                                                 planner=POSTPROCESS_PLANNER,
                                                 max_time=max_time - elapsed_time(start_time))
            if (new_layer_plan is not None) and (compute_duration(new_layer_plan) < compute_duration(layer_plan)):
                layer_plan = new_layer_plan
            user_input('{:.3f}->{:.3f}'.format(compute_duration(layer_plan), compute_duration(new_layer_plan)))

        # TODO: replan in a cost sensitive way
        layer_plan = apply_start(layer_plan, makespan)
        duration = compute_duration(layer_plan)
        makespan += duration
        print('\nLength: {} | Start: {:.3f} | End: {:.3f} | Duration: {:.3f} | Makespan: {:.3f}'.format(
            len(layer_plan), compute_start(layer_plan), compute_end(layer_plan), duration, makespan))
        full_plan.extend(layer_plan)
        removed.update(remaining)
    print(SEPARATOR)
    print_plan(full_plan)
    return full_plan, None

##################################################

def stripstream(robot1, obstacles, node_points, element_bodies, ground_nodes,
                dual=True, serialize=False, hierarchy=False, **kwargs):
    robots = mirror_robot(robot1, node_points) if dual else [robot1]
    elements = set(element_bodies)
    initial_confs = {ROBOT_TEMPLATE.format(i): Conf(robot) for i, robot in enumerate(robots)}
    saver = WorldSaver()

    layer_from_n = compute_layer_from_vertex(elements, node_points, ground_nodes)
    #layer_from_n = cluster_vertices(elements, node_points, ground_nodes) # TODO: increase resolution for small structures
    # TODO: compute directions from first, layer from second
    max_layer = max(layer_from_n.values())
    print('Max layer: {}'.format(max_layer))

    data = {}
    if serialize:
        plan, certificate = solve_serialized(robots, obstacles, node_points, element_bodies,
                                             ground_nodes, layer_from_n, initial_confs=initial_confs, **kwargs)
    else:
        plan, certificate = solve_joint(robots, obstacles, node_points, element_bodies,
                                        ground_nodes, layer_from_n, initial_confs=initial_confs, **kwargs)
    if plan is None:
        return None, data

    if hierarchy:
        print(SEPARATOR)
        static_facts = extract_static_facts(plan, certificate, initial_confs)
        partial_orders = compute_total_orders(plan)
        plan, certificate = solve_joint(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                                        initial_confs=initial_confs, can_print=False, can_transit=True,
                                        additional_init=static_facts, additional_orders=partial_orders, **kwargs)
        if plan is None:
            return None, data

    if plan and not isinstance(plan[0], DurativeAction):
        time_from_start = 0.
        retimed_plan = []
        for name, args in plan:
            command = args[-1]
            command.retime(start_time=time_from_start)
            retimed_plan.append(DurativeAction(name, args, time_from_start, command.duration))
            time_from_start += command.duration
        plan = retimed_plan
    plan = reverse_plan(plan)
    print('\nLength: {} | Makespan: {:.3f}'.format(len(plan), compute_duration(plan)))
    # TODO: retime using the TFD duration
    # TODO: attempt to resolve once without any optimistic facts to see if a solution exists
    # TODO: choose a better initial config
    # TODO: decompose into layers hierarchically

    #planned_elements = [args[2] for name, args, _, _ in sorted(plan, key=lambda a: get_end(a))] # TODO: remove approach
    #if not check_plan(extrusion_path, planned_elements):
    #    return None, data

    if has_gui():
        saver.restore()
        #label_nodes(node_points)
        # commands = [action.args[-1] for action in reversed(plan) if action.name == 'print']
        # trajectories = flatten_commands(commands)
        # elements = recover_sequence(trajectories)
        # draw_ordered(elements, node_points)
        # wait_if_gui('Continue?')

        #simulate_printing(node_points, trajectories)
        #display_trajectories(node_points, ground_nodes, trajectories)
        simulate_parallel(robots, plan)

    return None, data
    #return trajectories, data

##################################################

def get_wild_move_gen_fn(robots, static_obstacles, element_bodies, partial_orders=set(), collisions=True, **kwargs):
    incoming_supporters, _ = neighbors_from_orders(partial_orders)

    def wild_gen_fn(name, conf1, conf2, *args):
        is_initial = (conf1.element is None) and (conf2.element is not None)
        is_goal = (conf1.element is not None) and (conf2.element is None)
        if is_initial:
            supporters = []
        elif is_goal:
            supporters = list(element_bodies)
        else:
            supporters = [conf1.element]  # TODO: can also do according to levels
            retrace_supporters(conf1.element, incoming_supporters, supporters)
        element_obstacles = {element_bodies[e] for e in supporters}
        obstacles = set(static_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()

        robot = index_from_name(robots, name)
        conf1.assign()
        joints = get_movable_joints(robot)
        # TODO: break into pieces at the furthest part from the structure

        weights = JOINT_WEIGHTS
        resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
        disabled_collisions = get_disabled_collisions(robot)
        #path = [conf1, conf2]
        path = plan_joint_motion(robot, joints, conf2.positions, obstacles=obstacles,
                                 self_collisions=SELF_COLLISIONS, disabled_collisions=disabled_collisions,
                                 weights=weights, resolutions=resolutions,
                                 restarts=3, iterations=100, smooth=100)
        if not path:
            return
        path = [conf1.positions] + path[1:-1] + [conf2.positions]
        traj = MotionTrajectory(robot, joints, path)
        command = Command([traj])
        edges = [
            (conf1, command, conf2),
            (conf2, command, conf1), # TODO: reverse
        ]
        outputs = []
        #outputs = [(command,)]
        facts = []
        for q1, cmd, q2 in edges:
            facts.extend([
                ('Traj', name, cmd),
                ('CTraj', name, cmd),
                ('MoveAction', name, q1, q2, cmd),
            ])
        yield WildOutput(outputs, facts)
    return wild_gen_fn

def get_wild_print_gen_fn(robots, static_obstacles, node_points, element_bodies, ground_nodes,
                          initial_confs={}, return_home=False, collisions=True, **kwargs):
    # TODO: could reuse end-effector trajectories
    # TODO: max distance from nearby
    gen_fn_from_robot = {robot: get_print_gen_fn(robot, static_obstacles, node_points, element_bodies, ground_nodes,
                                                 p_nearby=1., approach_distance=0.05,
                                                 precompute_collisions=True, **kwargs) for robot in robots}
    wild_move_fn = get_wild_move_gen_fn(robots, static_obstacles, element_bodies, **kwargs)

    def wild_gen_fn(name, node1, element, node2):
        # TODO: could cache this
        # sequence = [result.get_mapping()['?e'].value for result in CURRENT_STREAM_PLAN]
        # index = sequence.index(element)
        # printed = sequence[:index]
        # TODO: this might need to be recomputed per iteration
        # TODO: condition on plan/downstream constraints
        # TODO: stream fusion
        # TODO: split element into several edges
        robot = index_from_name(robots, name)
        q0 = initial_confs[name]
        #generator = gen_fn_from_robot[robot](node1, element)
        for print_cmd, in gen_fn_from_robot[robot](node1, element):
            # TODO: need to merge safe print_cmd.colliding
            q1 = Conf(robot, print_cmd.start_conf, node=node1, element=element)
            q2 = Conf(robot, print_cmd.end_conf, node=node2, element=element)

            if return_home:
                # TODO: can decompose into individual movements as well
                output1 = next(wild_move_fn(name, q0, q1), None)
                if not output1:
                    continue
                transit_cmd1 = output1.values[0][0]
                print_cmd.trajectories = transit_cmd1.trajectories + print_cmd.trajectories
                output2 = next(wild_move_fn(name, q2, q0), None)
                if not output2:
                    continue
                transit_cmd2 = output2.values[0][0]
                print_cmd.trajectories = print_cmd.trajectories + transit_cmd2.trajectories
                q1 = q2 = q0 # TODO: must assert that AtConf holds

            outputs = [(q1, q2, print_cmd)]
            # Prevents premature collision checks
            facts = [('CTraj', name, print_cmd)] # + [('Dummy',)] # To force to be wild
            if collisions:
                facts.extend(('Collision', print_cmd, e2) for e2 in print_cmd.colliding)
            yield WildOutput(outputs,  facts)
    return wild_gen_fn

def get_collision_test(robots, collisions=True, **kwargs):
    # TODO: check end-effector collisions first
    def test(name1, command1, name2, command2):
        robot1, robot2 = index_from_name(robots, name1), index_from_name(robots, name2)
        if (robot1 == robot2) or not collisions:
            return False
        # TODO: check collisions between pairs of inflated adjacent element
        for traj1, traj2 in randomize(product(command1.trajectories, command2.trajectories)):
            # TODO: use swept aabbs for element checks
            aabbs1, aabbs2 = traj1.get_aabbs(), traj2.get_aabbs()
            swept_aabbs1 = {link: aabb_union(link_aabbs[link] for link_aabbs in aabbs1) for link in aabbs1[0]}
            swept_aabbs2 = {link: aabb_union(link_aabbs[link] for link_aabbs in aabbs2) for link in aabbs2[0]}
            swept_overlap = [(link1, link2) for link1, link2 in product(swept_aabbs1, swept_aabbs2)
                             if aabb_overlap(swept_aabbs1[link1], swept_aabbs2[link2])]
            if not swept_overlap:
                continue
            # for l1 in set(map(itemgetter(0), swept_overlap)):
            #     draw_aabb(swept_aabbs1[l1], color=RED)
            # for l2 in set(map(itemgetter(1), swept_overlap)):
            #     draw_aabb(swept_aabbs2[l2], color=BLUE)

            for index1, index2 in product(randomize(range(len(traj1.path))), randomize(range(len(traj2.path)))):
                overlap = [(link1, link2) for link1, link2 in swept_overlap
                           if aabb_overlap(aabbs1[index1][link1], aabbs2[index2][link2])]
                #overlap = list(product(aabbs1[index1], aabbs2[index2]))
                if not overlap:
                    continue
                set_configuration(robot1, traj1.path[index1])
                set_configuration(robot2, traj2.path[index2])
                #wait_if_gui()
                #if pairwise_collision(robot1, robot2):
                #    return True
                for link1, link2 in overlap:
                    if pairwise_link_collision(robot1, link1, robot2, link2):
                        #wait_if_gui()
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