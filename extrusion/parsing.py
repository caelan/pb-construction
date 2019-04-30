import json
import os

import numpy as np

from collections import namedtuple, OrderedDict
from examples.pybullet.utils.pybullet_tools.utils import add_line, create_cylinder, set_point, set_quat, \
    quat_from_euler, Euler
from extrusion.utils import is_ground

Element = namedtuple('Element', ['id', 'layer', 'nodes'])

EXTRUSION_DIRECTORY = os.path.join('..', 'assembly_instances', 'extrusion')
EXTRUSION_FILENAMES = {
    'djmm_bridge':      'DJMM_bridge.json',
    'djmm_test_block':  'djmm_test_block_S1_03-14-2019_w_layer.json',
    'mars_bubble':      'mars_bubble_S1_03-14-2019_w_layer.json',
    'sig_artopt-bunny': 'sig_artopt-bunny_S1_03-14-2019_w_layer.json',
    'topopt-100':       'topopt-100_S1_03-14-2019_w_layer.json',
    'topopt-205':       'topopt-205_S0.7_03-14-2019_w_layer.json',
    'topopt-310':       'topopt-310_S1_03-14-2019_w_layer.json',
    'voronoi':          'voronoi_S1_03-14-2019_w_layer.json',
    'simple_frame':     'simple_frame.json',
    'four-frame':       'four-frame.json',
}
DEFAULT_SCALE = 1e-3 # TODO: load different scales

def get_extrusion_path(extrusion_name):
    if extrusion_name not in EXTRUSION_FILENAMES:
        raise ValueError(extrusion_name)
    root_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root_directory, EXTRUSION_DIRECTORY, EXTRUSION_FILENAMES[extrusion_name])

def load_extrusion(extrusion_name):
    extrusion_path = get_extrusion_path(extrusion_name)
    print('Name: {}'.format(extrusion_name))
    print('Path: {}'.format(extrusion_path))
    with open(extrusion_path, 'r') as f:
        json_data = json.loads(f.read())

    elements = parse_elements(json_data)
    element_from_id = OrderedDict((element.id, element.nodes) for element in elements)
    node_points = parse_node_points(json_data)
    ground_nodes = parse_ground_nodes(json_data)
    print('Assembly: {} | Model: {} | Unit: {}'.format(
        json_data['assembly_type'], json_data['model_type'], json_data['unit'])) # extrusion, spatial_frame, millimeter
    print('Nodes: {} | Ground: {} | Elements: {}'.format(
        len(node_points), len(ground_nodes), len(elements)))
    return element_from_id, node_points, ground_nodes


def parse_point(json_point, scale=DEFAULT_SCALE):
    return scale * np.array([json_point['X'], json_point['Y'], json_point['Z']])


def parse_transform(json_transform):
    transform = np.eye(4)
    transform[:3, 3] = parse_point(json_transform['Origin']) # Normal
    transform[:3, :3] = np.vstack([parse_point(json_transform[axis], scale=1)
                                   for axis in ['XAxis', 'YAxis', 'ZAxis']])
    return transform


def parse_origin(json_data):
    return parse_point(json_data['base_frame_in_rob_base']['Origin'])


def parse_elements(json_data):
    return [Element(json_element.get('element_id', i), json_element.get('layer', None), tuple(json_element['end_node_ids']))
            for i, json_element in enumerate(json_data['element_list'])] # 'layer_id


def parse_node_points(json_data):
    origin = parse_origin(json_data)
    return [origin + parse_point(json_node['point']) for json_node in json_data['node_list']]


def parse_ground_nodes(json_data):
    return {i for i, json_node in enumerate(json_data['node_list']) if json_node['is_grounded'] == 1}

##################################################

def create_elements(node_points, elements, color=(1, 0, 0, 1)):
    # TODO: just shrink the structure to prevent worrying about collisions at end-points
    #radius = 0.0001
    #radius = 0.00005
    #radius = 0.000001
    radius = 1e-6
    # TODO: seems to be a min radius

    shrink = 0.01
    #shrink = 0.005
    #shrink = 0.002
    #shrink = 0.
    element_bodies = []
    for (n1, n2) in elements:
        p1, p2 = node_points[n1], node_points[n2]
        height = max(np.linalg.norm(p2 - p1) - 2*shrink, 0)
        #if height == 0: # Cannot keep this here
        #    continue
        center = (p1 + p2) / 2
        # extents = (p2 - p1) / 2
        body = create_cylinder(radius, height, color=color)
        set_point(body, center)
        element_bodies.append(body)

        delta = p2 - p1
        x, y, z = delta
        phi = np.math.atan2(y, x)
        theta = np.math.acos(z / np.linalg.norm(delta))
        set_quat(body, quat_from_euler(Euler(pitch=theta, yaw=phi)))
        # p1 is z=-height/2, p2 is z=+height/2
    return element_bodies

##################################################

def draw_element(node_points, element, color=(1, 0, 0)):
    n1, n2 = element
    p1 = node_points[n1]
    p2 = node_points[n2]
    return add_line(p1, p2, color=color[:3])


def draw_model(elements, node_points, ground_nodes):
    handles = []
    for element in elements:
        color = (0, 0, 1) if is_ground(element, ground_nodes) else (1, 0, 0)
        handles.append(draw_element(node_points, element, color=color))
    return handles
