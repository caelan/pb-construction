from collections import defaultdict, Counter
from itertools import product, permutations

import numpy as np

from extrusion.heuristics import compute_distance_from_node
from extrusion.stream import APPROACH_DISTANCE, SELF_COLLISIONS, get_print_gen_fn

from extrusion.utils import get_other_node, get_node_neighbors, get_midpoint, get_element_length, load_robot, \
    PrintTrajectory, retrace_supporters, JOINT_WEIGHTS, RESOLUTION, get_disabled_collisions, MotionTrajectory, Command
from extrusion.visualization import set_extrusion_camera
from pddlstream.language.stream import WildOutput
from pddlstream.language.temporal import compute_duration, create_planner
from pddlstream.utils import inclusive_range, neighbors_from_orders
from pybullet_tools.utils import wait_if_gui, VideoSaver, set_configuration, wait_for_duration, INF, get_pose, \
    point_from_pose, tform_point, invert, get_yaw, Pose, get_point, set_pose, Point, Euler, draw_pose, add_line, RED, \
    get_movable_joints, plan_joint_motion, randomize, aabb_union, aabb_overlap, pairwise_link_collision, \
    get_configuration

ROBOT_TEMPLATE = 'r{}'
DUAL_CONF = [np.pi/4, -np.pi/4, np.pi/2, 0, np.pi/4, -np.pi/2] # np.pi/8

GREEDY_PLANNER = create_planner(greedy=False, lazy=True, h_cea=True)
POSTPROCESS_PLANNER = create_planner(anytime=True, greedy=False, lazy=True, h_cea=False, h_makespan=True)
#POSTPROCESS_PLANNER = create_planner(anytime=True, lazy=False, h_makespan=True)

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

def extract_parallel_trajectories(plan):
    if plan is None:
        return None
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
    return trajectories


def simulate_parallel(robots, plan, time_step=0.1, speed_up=10., record=None): # None | video.mp4
    # TODO: ensure the step size is appropriate
    makespan = compute_duration(plan)
    print('\nMakespan: {:.3f}'.format(makespan))
    trajectories = extract_parallel_trajectories(plan)
    if trajectories is None:
        return
    wait_if_gui('Begin?')
    num_motion = sum(action.name == 'move' for action in plan)
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

##################################################

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
        #generator = gen_fn_from_robot[robot](node1, element)
        for print_cmd, in gen_fn_from_robot[robot](node1, element):
            # TODO: need to merge safe print_cmd.colliding
            q1 = Conf(robot, print_cmd.start_conf, node=node1, element=element)
            q2 = Conf(robot, print_cmd.end_conf, node=node2, element=element)

            if return_home:
                q0 = initial_confs[name]
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
