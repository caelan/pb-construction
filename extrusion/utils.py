from __future__ import print_function

import os
import random
import numpy as np

from collections import defaultdict

from examples.pybullet.utils.pybullet_tools.utils import set_point, Euler, get_movable_joints, set_joint_positions, \
    pairwise_collision, Pose, multiply, Point, load_model, \
    HideOutput, load_pybullet, link_from_name, has_link, joint_from_name, angle_between, set_pose

KUKA_PATH = '../conrob_pybullet/models/kuka_kr6_r900/urdf/kuka_kr6_r900_extrusion.urdf'
TOOL_NAME = 'eef_tcp_frame'
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

##################################################

def check_command_collision(tool_body, tool_from_root, command, bodies):
    # TODO: each new addition makes collision checking more expensive
    #offset = 4
    #for robot_conf in trajectory[offset:-offset]:
    collisions = [False for _ in range(len(bodies))]

    # TODO: separate into another method. Sort paths by tool poses first
    for trajectory in command.trajectories:
        indices = list(range(len(trajectory.path)))
        random.shuffle(indices)  # TODO: bisect
        for k in indices:
            tool_pose = trajectory.tool_path[k]
            set_pose(tool_body, multiply(tool_pose, tool_from_root))
            for i, body in enumerate(bodies):
                if not collisions[i]:
                    collisions[i] |= pairwise_collision(tool_body, body)
        for k in indices:
            robot_conf = trajectory.path[k]
            set_joint_positions(trajectory.robot, trajectory.joints, robot_conf)
            for i, body in enumerate(bodies):
                if not collisions[i]:
                    collisions[i] |= pairwise_collision(trajectory.robot, body)
    return collisions

#def get_grasp_rotation(direction, angle):
    #return Pose(euler=Euler(roll=np.pi / 2, pitch=direction, yaw=angle))
    #rot = Pose(euler=Euler(roll=np.pi / 2))
    #thing = (unit_point(), quat_from_vector_angle(direction, angle))
    #return multiply(thing, rot)

def sample_direction():
    ##roll = random.uniform(0, np.pi)
    #roll = np.pi/4
    #pitch = random.uniform(0, 2*np.pi)
    #return Pose(euler=Euler(roll=np.pi / 2 + roll, pitch=pitch))
    roll = random.uniform(-np.pi/2, np.pi/2)
    pitch = random.uniform(-np.pi/2, np.pi/2)
    return Pose(euler=Euler(roll=roll, pitch=pitch))


def get_grasp_pose(translation, direction, angle, reverse, offset=1e-3):
    #direction = Pose(euler=Euler(roll=np.pi / 2, pitch=direction))
    return multiply(Pose(point=Point(z=offset)),
                    Pose(euler=Euler(yaw=angle)),
                    direction,
                    Pose(point=Point(z=translation)),
                    Pose(euler=Euler(roll=(1-reverse) * np.pi)))


def load_world():
    root_directory = os.path.dirname(os.path.abspath(__file__))
    with HideOutput():
        floor = load_model('models/short_floor.urdf')
        robot = load_pybullet(os.path.join(root_directory, KUKA_PATH), fixed_base=True)
    set_point(floor, Point(z=-0.01))
    return floor, robot


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

##################################################

class MotionTrajectory(object):
    def __init__(self, robot, joints, path, attachments=[]):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.attachments = attachments
    def reverse(self):
        return self.__class__(self.robot, self.joints, self.path[::-1], self.attachments)
    def iterate(self):
        for conf in self.path[1:]:
            set_joint_positions(self.robot, self.joints, conf)
            yield
    def __repr__(self):
        return 'm({},{})'.format(len(self.joints), len(self.path))


class PrintTrajectory(object):
    def __init__(self, robot, joints, path, tool_path, element, reverse):
        self.robot = robot
        self.joints = joints
        self.path = path
        self.tool_path = tool_path
        assert len(self.path) == len(self.tool_path)
        self.n1, self.n2 = reversed(element) if reverse else element
        self.element = element
    def __repr__(self):
        return '{}->{}'.format(self.n1, self.n2)

class Command(object):
    def __init__(self, trajectories=[], colliding=set()):
        self.trajectories = tuple(trajectories)
        self.colliding = set(colliding)
    def reverse(self):
        return self.__class__([traj.reverse() for traj in reversed(self.trajectories)],
                              colliding=self.colliding)
    def iterate(self):
        for trajectory in self.trajectories:
            for output in trajectory.iterate():
                yield output
    def __repr__(self):
        return 'c[{}]'.format(','.format(map(str, self.trajectories)))

##################################################

def is_start_node(n1, e, node_points):
    return not element_supports(e, n1, node_points)

def doubly_printable(e, node_points):
    return all(is_start_node(n, e, node_points) for n in e)

def get_other_node(node1, element):
    assert node1 in element
    return element[node1 == element[0]]

def is_ground(element, ground_nodes):
    return any(n in ground_nodes for n in element)

##################################################

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

def downsample_nodes(elements, node_points, ground_nodes, n=None):
    node_order = list(range(len(node_points)))
    # np.random.shuffle(node_order)
    node_order = sorted(node_order, key=lambda n: node_points[n][2])
    elements = sorted(elements, key=lambda e: min(node_points[n][2] for n in e))

    if n is not None:
        node_order = node_order[:n]
    ground_nodes = [n for n in ground_nodes if n in node_order]
    elements = [element for element in elements
                if all(n in node_order for n in element)]
    return elements, ground_nodes