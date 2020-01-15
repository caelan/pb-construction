from __future__ import print_function

import os
import numpy as np

from collections import defaultdict, deque, namedtuple

from pyconmech import StiffnessChecker

from pybullet_tools.utils import get_link_pose, BodySaver, set_point, multiply, set_pose, set_joint_positions, \
    Point, load_model, HideOutput, load_pybullet, link_from_name, has_link, joint_from_name, angle_between, get_aabb, \
    get_distance, get_relative_pose, get_link_subtree, clone_body, randomize, pairwise_collision, wait_for_user, \
    get_movable_joints, get_all_links, get_bodies_in_region, pairwise_link_collision, draw_aabb, set_static, set_all_static
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
JOINT_WEIGHTS = np.array([0.3078557810844393, 0.443600199302506, 0.23544367607317915,
                          0.03637161028426032, 0.04644626184081511, 0.015054267683041092])

TRANS_TOL = 0.0015
ROT_TOL = 5 * np.pi / 180

INITIAL_CONF = [0, -np.pi/4, np.pi/4, 0, 0, 0]

##################################################

def load_world(use_floor=USE_FLOOR):
    root_directory = os.path.dirname(os.path.abspath(__file__))
    obstacles = []
    with HideOutput():
        robot = load_pybullet(os.path.join(root_directory, KUKA_PATH), fixed_base=True)
        set_static(robot)
        set_joint_positions(robot, get_movable_joints(robot), INITIAL_CONF)
        lower, _ = get_aabb(robot)
        if use_floor:
            floor = load_model('models/short_floor.urdf', fixed_base=True)
            obstacles.append(floor)
            #set_point(floor, Point(z=lower[2]))
            set_point(floor, Point(x=1.2, z=0.023-0.025))
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
    def get_link_path(self, link_name=TOOL_LINK):
        link = link_from_name(self.robot, link_name)
        if link not in self.path_from_link:
            with BodySaver(self.robot):
                self.path_from_link[link] = []
                for conf in self.path:
                    set_joint_positions(self.robot, self.joints, conf)
                    self.path_from_link[link].append(get_link_pose(self.robot, link))
        return self.path_from_link[link]
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

class PrintTrajectory(Trajectory):
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
        return '{}->{}'.format(self.n1, self.n2)

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
            for robot_conf in randomize(trajectory.path):
                set_joint_positions(trajectory.robot, trajectory.joints, robot_conf)
                for element in unknown_elements:
                    safe = not pairwise_collision(trajectory.robot, element_bodies[element])
                    if not safe:
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

def is_start_node(n1, e, node_points):
    return not element_supports(e, n1, node_points)

def doubly_printable(e, node_points):
    return all(is_start_node(n, e, node_points) for n in e)

def get_other_node(node1, element):
    assert node1 in element
    return element[node1 == element[0]]

def is_printable(element, printed_nodes):
    return any(n in printed_nodes for n in element)

def is_ground(element, ground_nodes):
    return is_printable(element, ground_nodes)

def get_ground_elements(elements, ground_nodes):
    return frozenset(filter(lambda e: is_ground(e, ground_nodes), elements))

def compute_element_distance(node_points, elements):
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

def create_stiffness_checker(extrusion_path, verbose=False):
    # TODO: the stiffness checker likely has a memory leak
    # https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
    if not os.path.exists(extrusion_path):
        raise FileNotFoundError(extrusion_path)
    with HideOutput():
        checker = StiffnessChecker(json_file_path=extrusion_path, verbose=verbose)
    #checker.set_output_json(True)
    #checker.set_output_json_path(file_path=os.getcwd(), file_name="stiffness-results.json")
    checker.set_self_weight_load(True)
    #checker.set_nodal_displacement_tol(transl_tol=0.005, rot_tol=10 * np.pi / 180)
    #checker.set_nodal_displacement_tol(transl_tol=0.003, rot_tol=5 * np.pi / 180)
    # checker.set_nodal_displacement_tol(transl_tol=1e-3, rot_tol=3 * (np.pi / 360))
    checker.set_nodal_displacement_tol(trans_tol=TRANS_TOL, rot_tol=ROT_TOL)
    #checker.set_loads(point_loads=None, include_self_weight=False, uniform_distributed_load={})
    return checker

def get_id_from_element(element_from_id):
    return {e: i for i, e in element_from_id.items()}

def get_extructed_ids(element_from_id, directed_elements):
    id_from_element = get_id_from_element(element_from_id)
    extruded_ids = []
    for directed in directed_elements:
        is_reverse = directed not in id_from_element
        assert (directed in id_from_element) != is_reverse
        element = directed[::-1] if is_reverse else directed
        extruded_ids.append(id_from_element[element])
    return sorted(extruded_ids)

Deformation = namedtuple('Deformation', ['success', 'displacements', 'fixities', 'reactions']) # TODO: get_max_nodal_deformation
Displacement = namedtuple('Displacement', ['dx', 'dy', 'dz', 'theta_x', 'theta_y', 'theta_z'])
Reaction = namedtuple('Reaction', ['fx', 'fy', 'fz', 'mx', 'my', 'mz'])

def force_from_reaction(reaction):
    return reaction[:3]

def torque_from_reaction(reaction):
    return reaction[3:]

##################################################

def evaluate_stiffness(extrusion_path, element_from_id, elements, checker=None, verbose=True):
    # TODO: check each component individually
    if not elements:
        return Deformation(True, {}, {}, {})
    #return True
    if checker is None:
        checker = create_stiffness_checker(extrusion_path, verbose=False)
    # TODO: load element_from_id
    extruded_ids = get_extructed_ids(element_from_id, elements)
    #print(checker.get_element_local2global_rot_matrices())
    #print(checker.get_element_stiffness_matrices(in_local_coordinate=False))

    #nodal_loads = checker.get_nodal_loads(existing_ids=[], dof_flattened=False) # per node
    #weight_loads = checker.get_self_weight_loads(existing_ids=[], dof_flattened=False) # get_nodal_loads = get_self_weight_loads?
    #for node in sorted(nodal_load):
    #    print(node, nodal_loads[node] - weight_loads[node])

    is_stiff = checker.solve(exist_element_ids=extruded_ids, if_cond_num=True)
    #print("has stored results: {0}".format(checker.has_stored_result()))
    success, nodal_displacement, fixities_reaction, element_reaction = checker.get_solved_results()
    assert is_stiff == success # TODO: this sometimes isn't true
    displacements = {i: Displacement(*d) for i, d in nodal_displacement.items()}
    fixities = {i: Reaction(*d) for i, d in fixities_reaction.items()}
    reactions = {i: (Reaction(*d[0]), Reaction(*d[1])) for i, d in element_reaction.items()}

    #translation = np.max(np.linalg.norm([d[:3] for d in displacements.values()], axis=1))
    #rotation = np.max(np.linalg.norm([d[3:] for d in displacements.values()], axis=1))

    #print("nodal displacement (m/rad):\n{0}".format(nodal_displacement)) # nodes x 7
    # TODO: investigate if nodal displacement can be used to select an ordering
    #print("fixities reaction (kN, kN-m):\n{0}".format(fixities_reaction)) # ground x 7
    #print("element reaction (kN, kN-m):\n{0}".format(element_reaction)) # elements x 13
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    max_trans, max_rot, max_trans_vid, max_rot_vid = checker.get_max_nodal_deformation()
    # The inverse of stiffness is flexibility or compliance
    if verbose:
        print('Stiff: {} | Compliance: {:.5f}'.format(is_stiff, checker.get_compliance()))
        print('Max translation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_trans, trans_tol, max_trans / trans_tol, max_trans_vid))
        print('Max rotation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_rot, rot_tol, max_rot / rot_tol, max_rot_vid))
    #disc = 10
    #exagg_ratio = 1.0
    #time_step = 1.0
    #orig_beam_shape = checker.get_original_shape(disc=disc, draw_full_shape=False)
    #beam_disp = checker.get_deformed_shape(exagg_ratio=exagg_ratio, disc=disc)
    return Deformation(is_stiff, displacements, fixities, reactions)

def test_stiffness(extrusion_path, element_from_id, elements, **kwargs):
    return evaluate_stiffness(extrusion_path, element_from_id, elements, **kwargs).success
