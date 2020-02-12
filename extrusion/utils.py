from __future__ import print_function

import os
import numpy as np
import math
import traceback

from collections import defaultdict, deque
from itertools import islice, cycle

from pybullet_tools.utils import get_link_pose, BodySaver, set_point, multiply, set_pose, set_joint_positions, \
    Point, HideOutput, load_pybullet, link_from_name, has_link, joint_from_name, angle_between, get_aabb, \
    get_distance, get_relative_pose, get_link_subtree, clone_body, randomize, get_movable_joints, get_all_links, get_bodies_in_region, pairwise_link_collision, \
    set_static, BASE_LINK, add_data_path, INF, load_model, create_plane, set_color, TAN, set_texture, create_box, \
    apply_alpha, point_from_pose, get_max_velocity, get_distance_fn
from pddlstream.utils import get_connected_components

KUKA_PATH = '../conrob_pybullet/models/kuka_kr6_r900/urdf/kuka_kr6_r900_extrusion.urdf'
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


INITIAL_CONF = [0, -np.pi/4, np.pi/4, 0, 0, 0]

#GROUND_COLOR = 0.9*np.ones(3)
GROUND_COLOR = 0.8*np.ones(3)
#GROUND_COLOR = TAN

##################################################

def get_pairs(sequence):
    return list(zip(sequence[:-1], sequence[1:]))

# https://docs.python.org/3.1/library/itertools.html#recipes
def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

##################################################

def load_world(use_floor=True):
    root_directory = os.path.dirname(os.path.abspath(__file__))
    obstacles = []
    #side, height = 10, 0.01
    with HideOutput():
        robot = load_pybullet(os.path.join(root_directory, KUKA_PATH), fixed_base=True)
        #print([get_max_velocity(robot, joint) for joint in get_movable_joints(robot)])
        set_static(robot)
        set_joint_positions(robot, get_movable_joints(robot), INITIAL_CONF)
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
    # TODO: always include ground nodes
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

def get_custom_limits(robot):
    return {joint_from_name(robot, joint): limits
            for joint, limits in CUSTOM_LIMITS.items()}

def get_cspace_distance(robot, q1, q2):
    #return get_distance(q1, q2)
    joints = get_movable_joints(robot)
    distance_fn = get_distance_fn(robot, joints, weights=JOINT_WEIGHTS)
    return distance_fn(q1, q2)

##################################################

class EndEffector(object):
    def __init__(self, robot, ee_link, tool_link, **kwargs):
        self.robot = robot
        self.ee_link = ee_link
        self.tool_link = tool_link
        self.tool_from_ee = get_relative_pose(self.robot, self.ee_link, self.tool_link)
        tool_links = get_link_subtree(robot, self.ee_link)
        self.body = clone_body(robot, links=tool_links, **kwargs)
        set_static(self.body)
        # for link in get_all_links(tool_body):
        #    set_color(tool_body, np.zeros(4), link)
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
            yield

class MotionTrajectory(Trajectory): # Transfer
    def __init__(self, robot, joints, path, attachments=[]):
        super(MotionTrajectory, self).__init__(robot, joints, path)
        self.attachments = attachments
    def reverse(self):
        return self.__class__(self.robot, self.joints, self.path[::-1], self.attachments)
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
    def __repr__(self):
        return 'p({}->{})'.format(self.n1, self.n2)

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

def compute_element_distance(node_points, elements):
    if not elements:
        return 0.
    return sum(get_distance(node_points[n1], node_points[n2]) for n1, n2 in elements)

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

def compute_sequence_distance(node_points, directed_elements, start=None):
    if directed_elements is None:
        return INF
    distance = 0.
    position = start
    for (n1, n2) in directed_elements:
        if position is not None:
            distance += get_distance(position, node_points[n1])
        distance += get_distance(node_points[n1], node_points[n2])
        position = node_points[n2]
    if start is not None:
        distance += get_distance(position, start)
    return distance

##################################################

def is_start_node(n1, e, node_points):
    return not element_supports(e, n1, node_points)

def doubly_printable(e, node_points):
    return all(is_start_node(n, e, node_points) for n in e)

def get_supported_orders(elements, node_points):
    node_neighbors = get_node_neighbors(elements)
    orders = set()
    for node in node_neighbors:
        supporters = {e for e in node_neighbors[node] if element_supports(e, node, node_points)}
        printers = {e for e in node_neighbors[node] if is_start_node(node, e, node_points)
                    and not doubly_printable(e, node_points)}
        orders.update((e1, e2) for e1 in supporters for e2 in printers)
    return orders

def element_supports(e, n1, node_points): # A property of nodes
    # TODO: support polygon (ZMP heuristic)
    # TODO: recursively apply as well
    # TODO: end-effector force
    # TODO: allow just a subset to support
    # TODO: construct using only upwards
    n2 = get_other_node(n1, e)
    delta = node_points[n2] - node_points[n1]
    theta = angle_between(delta, [0, 0, -1])
    return theta < (np.pi / 2 - SUPPORT_THETA)

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

BYTES_PER_KILOBYTE = math.pow(2, 10)
BYTES_PER_GIGABYTE = math.pow(2, 30)
KILOBYTES_PER_GIGABYTE = BYTES_PER_GIGABYTE / BYTES_PER_KILOBYTE

MAX_MEMORY = INF
#MAX_MEMORY = 1.5 * KILOBYTES_PER_GIGABYTE # 1.5 GB

def get_memory_in_kb():
    # https://pypi.org/project/psutil/
    # https://psutil.readthedocs.io/en/latest/
    import psutil
    #rss: aka "Resident Set Size", this is the non-swapped physical memory a process has used. (bytes)
    #vms: aka "Virtual Memory Size", this is the total amount of virtual memory used by the process. (bytes)
    #shared: (Linux) memory that could be potentially shared with other processes.
    #text (Linux, BSD): aka TRS (text resident set) the amount of memory devoted to executable code.
    #data (Linux, BSD): aka DRS (data resident set) the amount of physical memory devoted to other than executable code.
    #lib (Linux): the memory used by shared libraries.
    #dirty (Linux): the number of dirty pages.
    #pfaults (macOS): number of page faults.
    #pageins (macOS): number of actual pageins.
    process = psutil.Process(os.getpid())
    #process.pid()
    #process.ppid()
    pmem = process.memory_info() # this seems to actually get the current memory!
    return pmem.vms / BYTES_PER_KILOBYTE
    #print(process.memory_full_info())
    #print(process.memory_percent())
    # process.rlimit(psutil.RLIMIT_NOFILE)  # set resource limits (Linux only)
    #print(psutil.virtual_memory())
    #print(psutil.swap_memory())
    #print(psutil.pids())

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

# https://www.jujens.eu/posts/en/2018/Jun/02/python-timeout-function/
# https://code-maven.com/python-timeout
# https://pypi.org/project/func-timeout/
# https://pypi.org/project/timeout-decorator/
# https://eli.thegreenplace.net/2011/08/22/how-not-to-set-a-timeout-on-a-computation-in-python
# https://docs.python.org/3/library/signal.html
# https://docs.python.org/3/library/contextlib.html
# https://stackoverflow.com/a/22348885

import signal
from contextlib import contextmanager

def raise_timeout(signum, frame):
    raise TimeoutError()

@contextmanager
def timeout(duration):
    assert 0 < duration
    if duration == INF:
        yield
        return
    # Register a function to raise a TimeoutError on the signal
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``duration``
    signal.alarm(int(math.ceil(duration)))
    try:
        yield
    except TimeoutError as e:
        print('Timeout after {} sec'.format(duration))
        #traceback.print_exc()
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
