from __future__ import print_function

import cProfile
import os
import pstats

import numpy as np

from collections import defaultdict, deque

from pybullet_tools.utils import get_link_pose, BodySaver, set_point, multiply, set_pose, set_joint_positions, \
    Point, HideOutput, load_pybullet, link_from_name, has_link, joint_from_name, get_aabb, \
    get_distance, get_relative_pose, get_link_subtree, clone_body, randomize, get_movable_joints, get_all_links, \
    waypoints_from_path, get_bodies_in_region, pairwise_link_collision, \
    set_static, BASE_LINK, INF, create_plane, apply_alpha, point_from_pose, get_distance_fn, get_memory_in_kb, \
    get_pairs, Saver, set_configuration, add_line, RED, aabb_union, TRANSPARENT, set_color, get_joint_names
from pddlstream.utils import get_connected_components

KUKA_DIR = '../conrob_pybullet/models/kuka_kr6_r900/urdf/'
#KUKA_PATH = 'kuka_kr6_r900_extrusion.urdf'
KUKA_PATH = 'kuka_kr6_r900_extrusion_simple.urdf'

TOOL_LINK = 'eef_tcp_frame'
EE_LINK = 'eef_base_link' # robot_tool0
# [u'base_frame_in_rob_base', u'element_list', u'node_list', u'assembly_type', u'model_type', u'unit']

# TODO: import from SRDF
DISABLED_COLLISIONS = [
    # ('robot_link_1', 'workspace_objects'),
    # ('robot_link_2', 'workspace_objects'),
    # ('robot_link_3', 'workspace_objects'),
    # ('robot_link_4', 'workspace_objects'),
    ('robot_link_5', 'eef_base_link'),
]
CUSTOM_LIMITS = {
    'robot_joint_a1': (-np.pi/2, np.pi/2),
}
SUPPORT_THETA = np.math.radians(10)  # Support polygon

USE_FLOOR = True

RESOLUTION = 0.005
# JOINT_WEIGHTS = np.array([0.3078557810844393, 0.443600199302506, 0.23544367607317915,
#                           0.03637161028426032, 0.04644626184081511, 0.015054267683041092])
JOINT_WEIGHTS = np.reciprocal([6.28318530718, 5.23598775598, 6.28318530718,
                               6.6497044501, 6.77187749774, 10.7337748998]) # sec / radian
# TODO: cumulative

INITIAL_CONF = [0, -np.pi/4, np.pi/4, 0, 0, 0]

TOOL_VELOCITY = 0.01 # m/s

#GROUND_COLOR = 0.9*np.ones(3)
GROUND_COLOR = 0.8*np.ones(3)
#GROUND_COLOR = TAN

##################################################

MAX_MEMORY = INF
#MAX_MEMORY = 1.5 * KILOBYTES_PER_GIGABYTE # 1.5 GB

def check_memory(max_memory=MAX_MEMORY):
    if max_memory == INF:
        return True
    memory_kb = get_memory_in_kb()
    #print('Peak memory: {} | Max memory: {}'.format(peak_memory, max_memory))
    if memory_kb <= max_memory:
        return True
    print('Memory of {:.0f} KB exceeds memory limit of {:.0f} KB'.format(memory_kb, max_memory))
    return False

##################################################

def load_robot():
    root_directory = os.path.dirname(os.path.abspath(__file__))
    kuka_path = os.path.join(root_directory, KUKA_DIR, KUKA_PATH)
    with HideOutput():
        robot = load_pybullet(kuka_path, fixed_base=True)
        #print([get_max_velocity(robot, joint) for joint in get_movable_joints(robot)])
        set_static(robot)
        set_configuration(robot, INITIAL_CONF)
    return robot

def load_world(use_floor=True):
    obstacles = []
    #side, height = 10, 0.01
    robot = load_robot()
    with HideOutput():
        lower, _ = get_aabb(robot)
        if use_floor:
            #floor = load_model('models/short_floor.urdf', fixed_base=True)
            #add_data_path()
            #floor = load_pybullet('plane.urdf', fixed_base=True)
            #set_color(floor, TAN)
            #floor = create_box(w=side, l=side, h=height, color=apply_alpha(GROUND_COLOR))
            floor = create_plane(color=apply_alpha(GROUND_COLOR))
            obstacles.append(floor)
            #set_point(floor, Point(z=lower[2]))
            set_point(floor, Point(x=1.2, z=0.023-0.025)) # -0.02
        else:
            floor = None # TODO: make this an empty list of obstacles
        #set_all_static()
    return obstacles, robot


def prune_dominated(trajectories):
    start_len = len(trajectories)
    for traj1 in list(trajectories):
        if any((traj1 != traj2) and (traj2.colliding <= traj1.colliding)
               for traj2 in trajectories):
            trajectories.remove(traj1)
    return len(trajectories) == start_len

##################################################

def get_node_neighbors(elements):
    node_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        node_neighbors[n1].add(e)
        node_neighbors[n2].add(e)
    return node_neighbors

def nodes_from_elements(elements):
    return {n for e in elements for n in e}

def get_element_neighbors(elements):
    node_neighbors = get_node_neighbors(elements)
    element_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        element_neighbors[e].update(node_neighbors[n1])
        element_neighbors[e].update(node_neighbors[n2])
        element_neighbors[e].remove(e)
    return element_neighbors

##################################################

def get_disabled_collisions(robot):
    return {tuple(link_from_name(robot, link)
                  for link in pair if has_link(robot, link))
                  for pair in DISABLED_COLLISIONS}

def get_custom_limits(robot, named_limits=CUSTOM_LIMITS):
    return {joint_from_name(robot, joint): limits
            for joint, limits in named_limits.items()}

def get_cspace_distance(robot, q1, q2):
    #return get_distance(q1, q2)
    joints = get_movable_joints(robot)
    distance_fn = get_distance_fn(robot, joints, weights=JOINT_WEIGHTS)
    return distance_fn(q1, q2)

def retime_waypoints(waypoints, start_time=0.):
    durations = [start_time] + [get_distance(*pair) / TOOL_VELOCITY for pair in get_pairs(waypoints)]
    return np.cumsum(durations)

##################################################

class EndEffector(object):
    def __init__(self, robot, ee_link, tool_link, color=TRANSPARENT, **kwargs):
        self.robot = robot
        self.ee_link = ee_link
        self.tool_link = tool_link
        self.tool_from_ee = get_relative_pose(self.robot, self.ee_link, self.tool_link)
        tool_links = get_link_subtree(robot, self.ee_link)
        self.body = clone_body(robot, links=tool_links, visual=False, collision=True, **kwargs)
        set_static(self.body)
        if color is not None:
            for link in get_all_links(self.body):
                set_color(self.body, color, link) # None = white
    def get_tool_pose(self):
        return get_link_pose(self.robot, self.tool_link)
    def set_pose(self, tool_pose):
        pose = multiply(tool_pose, self.tool_from_ee)
        set_pose(self.body, pose)
        return pose
    @property
    def tool_from_root(self):
        return self.tool_from_ee
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.robot, self.body)

##################################################

class Trajectory(object):
    def __init__(self, robot, joints, path):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.path_from_link = {}
        self.intersecting = []
        self.spline = None
        self.aabbs = None
    @property
    def start_conf(self):
        if not self.path:
            return None
        return self.path[0]
    @property
    def end_conf(self):
        if not self.path:
            return None
        return self.path[-1]
    def get_intersecting(self):
        if self.intersecting:
            return self.intersecting
        robot_links = get_all_links(self.robot)
        # TODO: might need call step_simulation
        with BodySaver(self.robot):
            for conf in self.path:
                set_joint_positions(self.robot, self.joints, conf)
                intersecting = {}
                for robot_link in robot_links:
                    for body, _ in get_bodies_in_region(get_aabb(self.robot, link=robot_link)):
                        if body != self.robot:
                            intersecting.setdefault(robot_link, set()).add(body)
                    #print(get_bodies_in_region(get_aabb(self.robot, robot_link)))
                    #draw_aabb(get_aabb(self.robot, robot_link))
                    #wait_for_user()
                self.intersecting.append(intersecting)
        return self.intersecting
    def get_link_path(self, link_name=TOOL_LINK):
        link = link_from_name(self.robot, link_name)
        if link not in self.path_from_link:
            with BodySaver(self.robot):
                self.path_from_link[link] = []
                for conf in self.path:
                    set_joint_positions(self.robot, self.joints, conf)
                    self.path_from_link[link].append(get_link_pose(self.robot, link))
        return self.path_from_link[link]
    def get_distance(self):
        if not self.path:
            return 0.
        return sum(get_cspace_distance(self.robot, *pair) for pair in get_pairs(self.path))
    def get_link_distance(self, **kwargs):
        # TODO: just endpoints
        link_path = list(map(point_from_pose, self.get_link_path(**kwargs)))
        return sum(get_distance(*pair) for pair in get_pairs(link_path))
    def __len__(self):
        return len(self.path)
    def reverse(self):
        raise NotImplementedError()
    def iterate(self):
        for conf in self.path[1:]:
            set_joint_positions(self.robot, self.joints, conf)
            yield conf
    def get_aabbs(self):
        #traj.aabb = aabb_union(map(get_turtle_traj_aabb, traj.iterate())) # TODO: union
        if self.aabbs is not None:
            return self.aabbs
        self.aabbs = []
        links = get_all_links(self.robot)
        with BodySaver(self.robot):
            for conf in self.path:
                set_joint_positions(self.robot, self.joints, conf)
                self.aabbs.append({link: get_aabb(self.robot, link) for link in links})
        return self.aabbs
    def retime(self, **kwargs):
        # TODO: could also retime using the given end time
        from scipy.interpolate import interp1d
        #if self.spline is not None:
        #    return self.spline
        #tool_path = self.tool_path
        tool_path = self.get_link_path()
        positions = list(map(point_from_pose, tool_path))
        times_from_start = retime_waypoints(positions, **kwargs)
        self.spline = interp1d(times_from_start, self.path, kind='linear', axis=0)
        return self.spline
    @property
    def start_time(self):
        return self.spline.x[0]
    @property
    def end_time(self):
        return self.spline.x[-1]
    @property
    def duration(self):
        return self.end_time - self.start_time
    def at(self, time_from_start):
        assert self.spline is not None
        if (time_from_start < self.start_time) or (self.end_time < time_from_start):
            return None
        conf = self.spline(time_from_start)
        set_joint_positions(self.robot, self.joints, conf)
        return conf
    def interpolate(self):
        # TODO: linear or spline interpolation
        raise NotImplementedError()
    def extract_data(self, **kwargs):
        return {
            'joints': get_joint_names(self.robot, self.joints),
            'waypoints': list(map(tuple, waypoints_from_path(self.path))),
        }

class MotionTrajectory(Trajectory): # Transfer
    def __init__(self, robot, joints, path, attachments=[]):
        # TODO: store the end effector
        super(MotionTrajectory, self).__init__(robot, joints, path)
        self.attachments = attachments
    def reverse(self):
        return self.__class__(self.robot, self.joints, self.path[::-1], self.attachments)
    def extract_data(self, **kwargs):
        data = {
            'mode': 'transit', # TODO: extract automatically
        }
        data.update(super(MotionTrajectory, self).extract_data(**kwargs))
        return data
    def __repr__(self):
        return 'm({},{})'.format(len(self.joints), len(self.path))

class PrintTrajectory(Trajectory): # TODO: add element body?
    def __init__(self, end_effector, joints, path, tool_path, element, is_reverse=False):
        super(PrintTrajectory, self).__init__(end_effector.robot, joints, path)
        self.end_effector = end_effector
        self.tool_path = tool_path
        self.is_reverse = is_reverse
        #assert len(self.path) == len(self.tool_path)
        self.element = element
        self.n1, self.n2 = reversed(element) if self.is_reverse else element
        self.last_point = None
        self.handles = []
    @property
    def directed_element(self):
        return (self.n1, self.n2)
    def get_link_path(self, link_name=TOOL_LINK):
        if link_name == TOOL_LINK:
            return self.tool_path
        return super(PrintTrajectory, self).get_link_path(link_name)
    def reverse(self):
        return self.__class__(self.end_effector, self.joints, self.path[::-1],
                              self.tool_path[::-1], self.element, not self.is_reverse)
    def extract_data(self, **kwargs):
        data = {
            'mode': 'print',
            'element': self.element,
            'node1': self.n1,
            'node2': self.n2,
        }
        data.update(super(PrintTrajectory, self).extract_data(**kwargs))
        return data
    def __repr__(self):
        return 'p({}->{})'.format(self.n1, self.n2)
    def at(self, time_from_start):
        current_conf = super(PrintTrajectory, self).at(time_from_start)
        if current_conf is None:
            if self.last_point is not None:
                #set_configuration(self.robot, INITIAL_CONF) # TODO: return here
                end_point = point_from_pose(self.tool_path[-1])
                self.handles.append(add_line(self.last_point, end_point, color=RED))
                self.last_point = None
        else:
            if self.last_point is None:
                self.last_point = point_from_pose(self.tool_path[0])
            current_point = point_from_pose(self.end_effector.get_tool_pose())
            self.handles.append(add_line(self.last_point, current_point, color=RED))
            self.last_point = current_point
        return current_conf
    def interpolate_tool(self, node_points, **kwargs):
        from scipy.interpolate import interp1d
        positions = [node_points[self.n1], node_points[self.n2]]
        #positions = list(map(point_from_pose, self.tool_path))
        times_from_start = retime_waypoints(positions, **kwargs)
        return interp1d(times_from_start, positions, kind='linear', axis=0)
    def interpolate(self):
        # TODO: maintain a constant end-effector velocity by retiming
        raise NotImplementedError()

##################################################

def get_print_distance(trajectories, teleport=False):
    if trajectories is None:
        return INF
    distance = 0.
    for trajectory in trajectories:
        if teleport and isinstance(trajectory, MotionTrajectory):
            distance += get_cspace_distance(trajectory.robot, trajectory.path[0], trajectory.path[-1])
        else:
            distance += trajectory.get_distance()
    return distance

def extract_plan_data(trajectories, **kwargs):
    if trajectories is None:
        return None
    plan = [trajectory.extract_data(**kwargs) for trajectory in trajectories]
    #for i, data in enumerate(plan):
    #    print(i, data)
    return plan

##################################################

def recover_sequence(plan):
    if plan is None:
        return plan
    return [traj.element for traj in plan if isinstance(traj, PrintTrajectory)]

def recover_directed_sequence(plan):
    if plan is None:
        return plan
    return [traj.directed_element for traj in plan if isinstance(traj, PrintTrajectory)]

def flatten_commands(commands):
    if commands is None:
        return None
    return [traj for command in commands for traj in command.trajectories]

##################################################

class Command(object):
    def __init__(self, trajectories=[], safe_per_element={}):
        self.trajectories = list(trajectories)
        self.safe_per_element = dict(safe_per_element)
        self.colliding = set()
    @property
    def print_trajectory(self):
        for traj in self.trajectories:
            if isinstance(traj, PrintTrajectory):
                return traj
        return None
    @property
    def start_conf(self):
        return self.trajectories[0].start_conf
    @property
    def end_conf(self):
        return self.trajectories[-1].end_conf
    @property
    def elements(self):
        return recover_sequence(self.trajectories)
    @property
    def directed_elements(self):
        return recover_directed_sequence(self.trajectories)
    def get_distance(self):
        return sum(traj.get_distance() for traj in self.trajectories)
    def get_link_distance(self, **kwargs):
        return sum(traj.get_link_distance(**kwargs) for traj in self.trajectories)
    def set_safe(self, element):
        assert self.safe_per_element.get(element, True) is True
        self.safe_per_element[element] = True
    def set_unsafe(self, element):
        assert self.safe_per_element.get(element, False) is False
        self.safe_per_element[element] = False
        self.colliding.add(element)
    def update_safe(self, elements):
        for element in elements:
            self.set_safe(element)
    def is_safe(self, elements, element_bodies):
        # TODO: check the end-effector first
        known_elements = set(self.safe_per_element) & set(elements)
        if not all(self.safe_per_element[e] for e in known_elements):
            return False
        unknown_elements = randomize(set(elements) - known_elements)
        if not unknown_elements:
            return True
        for trajectory in randomize(self.trajectories): # TODO: could cache each individual collision
            intersecting = trajectory.get_intersecting()
            for i in randomize(range(len(trajectory))):
                set_joint_positions(trajectory.robot, trajectory.joints, trajectory.path[i])
                for element in unknown_elements:
                    body = element_bodies[element]
                    #if not pairwise_collision(trajectory.robot, body):
                    #    self.set_unsafe(element)
                    #    return False
                    for robot_link, bodies in intersecting[i].items():
                        #print(robot_link, bodies, len(bodies))
                        if (element_bodies[element] in bodies) and pairwise_link_collision(
                                trajectory.robot, robot_link, body, link2=BASE_LINK):
                            self.set_unsafe(element)
                            return False
        self.update_safe(elements)
        return True
    def reverse(self):
        return self.__class__([traj.reverse() for traj in reversed(self.trajectories)],
                              safe_per_element=self.safe_per_element)
    def iterate(self):
        for trajectory in self.trajectories:
            for output in trajectory.iterate():
                yield output
    @property
    def start_time(self):
        return self.trajectories[0].start_time
    @property
    def end_time(self):
        return self.trajectories[-1].end_time
    @property
    def duration(self):
        return self.end_time - self.start_time
    def retime(self, start_time=0, **kwargs):
        for traj in self.trajectories:
            traj.retime(start_time=start_time)
            start_time += traj.duration
    def __repr__(self):
        return 'c[{}]'.format(','.join(map(repr, self.trajectories)))

##################################################

def is_start(node1, element):
    assert node1 in element
    return node1 == element[0]

def is_end(node2, element):
    assert node2 in element
    return node2 == element[1]

def reverse_element(element):
    return element[::-1]

def is_reversed(all_elements, element):
    assert (element in all_elements) != (reverse_element(element) in all_elements)
    return element not in all_elements

def get_undirected(all_elements, directed):
    is_reverse = is_reversed(all_elements, directed)
    assert (directed in all_elements) != is_reverse
    return reverse_element(directed) if is_reverse else directed

def get_directions(element):
    return {element, reverse_element(element)}

def get_other_node(node1, element):
    assert node1 in element
    return element[node1 == element[0]]

def is_printable(element, printed_nodes):
    return any(n in printed_nodes for n in element)

def is_ground(element, ground_nodes):
    return is_printable(element, ground_nodes)

def get_ground_elements(all_elements, ground_nodes):
    return frozenset(filter(lambda e: is_ground(e, ground_nodes), all_elements))

def get_element_length(element, node_points):
    n1, n2 = element
    return get_distance(node_points[n2], node_points[n1])

def compute_element_distance(node_points, elements):
    if not elements:
        return 0.
    return sum(get_element_length(element, node_points) for element in elements)

def compute_printed_nodes(ground_nodes, printed):
    return nodes_from_elements(printed) | set(ground_nodes)

def compute_printable_elements(all_elements, ground_nodes, printed):
    ground_elements = get_ground_elements(all_elements, ground_nodes)
    if ground_elements <= printed:
        nodes = compute_printed_nodes(ground_nodes, printed)
    else:
        nodes = ground_nodes
    return {element for element in set(all_elements) - set(printed)
            if is_printable(element, nodes)}

def compute_printable_directed(all_elements, ground_nodes, printed):
    nodes = compute_printed_nodes(ground_nodes, printed)
    for element in set(all_elements) - printed:
        for directed in get_directions(element):
            node1, node2 = directed
            if node1 in nodes:
                yield directed

def get_midpoint(node_points, element):
    return np.average([node_points[n] for n in element], axis=0)

##################################################

def compute_transit_distance(node_points, directed_elements, start=None, end=None):
    if directed_elements is None:
        return INF
    assert directed_elements
    # Could instead compute full distance and subtract
    pairs = []
    if start is not None:
        pairs.append((start, node_points[directed_elements[0][0]]))
    pairs.extend((node_points[directed1[1]], node_points[directed2[0]])
                 for directed1, directed2 in get_pairs(directed_elements))
    if end is not None:
        pairs.append((node_points[directed_elements[-1][1]], end))
    return sum(get_distance(*pair) for pair in pairs)

def compute_sequence_distance(node_points, directed_elements, start=None, end=None):
    if directed_elements is None:
        return INF
    distance = 0.
    position = start
    for (n1, n2) in directed_elements:
        if position is not None:
            distance += get_distance(position, node_points[n1])
        distance += get_distance(node_points[n1], node_points[n2])
        position = node_points[n2]
    if end is not None:
        distance += get_distance(position, end)
    return distance

##################################################

def retrace_supporters(element, incoming_edges, supporters):
    for element2 in incoming_edges[element]:
        if element2 not in supporters:
            retrace_supporters(element2, incoming_edges, supporters=supporters)
            supporters.append(element2)

##################################################

def downselect_elements(elements, nodes):
    return [element for element in elements if all(n in nodes for n in element)]

def downsample_nodes(elements, node_points, ground_nodes, num=None):
    if num is None:
        return elements, ground_nodes
    node_order = list(range(len(node_points)))
    # np.random.shuffle(node_order)
    node_order = sorted(node_order, key=lambda n: node_points[n][2])
    elements = sorted(elements, key=lambda e: min(node_points[n][2] for n in e))

    if num is not None:
        node_order = node_order[:num]
    ground_nodes = [n for n in ground_nodes if n in node_order]
    return downselect_elements(elements, node_order), ground_nodes

##################################################

def check_connected(ground_nodes, printed_elements):
    # TODO: could merge with my connected components algorithm
    if not printed_elements:
        return True
    node_neighbors = get_node_neighbors(printed_elements)
    queue = deque(ground_nodes)
    visited_nodes = set(ground_nodes)
    visited_elements = set()
    while queue:
        node1 = queue.popleft()
        for element in node_neighbors[node1]:
            visited_elements.add(element)
            node2 = get_other_node(node1, element)
            if node2 not in visited_nodes:
                queue.append(node2)
                visited_nodes.add(node2)
    return printed_elements <= visited_elements

def get_connected_structures(elements):
    edges = {(e1, e2) for e1, neighbors in get_element_neighbors(elements).items()
             for e2 in neighbors}
    return get_connected_components(elements, edges)

##################################################

def get_id_from_element(element_from_id):
    return {e: i for i, e in element_from_id.items()}

def get_extructed_ids(element_from_id, directed_elements):
    id_from_element = get_id_from_element(element_from_id)
    extruded_ids = []
    for directed in directed_elements:
        element = get_undirected(id_from_element, directed)
        extruded_ids.append(id_from_element[element])
    return sorted(extruded_ids)

def compute_z_distance(node_points, element):
    # Distance to a ground plane
    # Opposing gravitational force
    return get_midpoint(node_points, element)[2]

##################################################

# TODO: from pddlstream.utils import Profiler
class Profiler(Saver):
    def __init__(self, cumulative=False, num=25):
        self.field = 'cumtime' if cumulative else 'tottime'
        self.num = num
        self.pr = cProfile.Profile()
        self.pr.enable()
    # def __enter__(self):
    #     return self # Enter called at with
    def restore(self):
        self.pr.disable()
        pstats.Stats(self.pr).sort_stats(self.field).print_stats(self.num)
