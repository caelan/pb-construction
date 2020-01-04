import json
import os
import numpy as np

from collections import namedtuple, OrderedDict
from pybullet_tools.utils import create_cylinder, set_point, set_quat, \
    quat_from_euler, Euler, tform_point, multiply, tform_from_pose, pose_from_tform, RED, apply_alpha

RADIUS = 1e-3
SHRINK = 0.003 # 0. | 0.002 | 0.005 | 0.003

Element = namedtuple('Element', ['id', 'layer', 'nodes'])

# https://github.com/yijiangh/assembly_instances/tree/master/extrusion
# EXTRUSION_DIRECTORY = os.path.join('..', 'assembly_instances', 'extrusion')
EXTRUSION_DIRECTORY = r'C:\Users\yijiangh\Documents\pb_ws\pychoreo\src\pychoreo_examples\data\assembly_instances\extrusion'
EXTRUSION_FILENAMES = {
    # bunny (3)
    'C': 'C_shape.json',
    'djmm_bridge': 'DJMM_bridge.json',
    'djmm_test_block': 'djmm_test_block_S1_03-14-2019_w_layer.json',
    'klein': 'klein_bottle.json',
    'knot': 'tre_foil_knot.json',
    'mars_bubble': 'mars_bubble_S1_03-14-2019_w_layer.json',
    'sig_artopt-bunny': 'sig_artopt-bunny_S1_03-14-2019_w_layer.json',
    'topopt-100': 'topopt-100_S1_03-14-2019_w_layer.json',
    'topopt-205': 'topopt-205_S0.7_03-14-2019_w_layer.json',
    'topopt-205_rotated': 'topopt-205_rotated.json',
    'topopt-310': 'topopt-310_S1_03-14-2019_w_layer.json',
    'voronoi': 'voronoi_S1_03-14-2019_w_layer.json',
    'topopt-101_tiny': 'topopt-101_tiny.json',
    'topopt-205_long_beam_test' : 'topopt-205_long_beam_test.json',
    'topopt-205_rotated_S1.5' : 'topopt-205_rotated_S1.5.json',
    'klein_bottle_trail' : 'klein_bottle_trail.json',
    'klein_bottle_trail_S2' : 'klein_bottle_trail_S2.json',
    'tre_foil_knot_S1.35' : 'tre_foil_knot_S1.35.json',
    'compas_fea_beam_tree_M_simp' : 'compas_fea_beam_tree_M_simp.json',
    'compas_fea_beam_tree_S_simp' : 'compas_fea_beam_tree_S_simp.json',
}
DEFAULT_SCALE = 1e-3 # TODO: load different scales

def get_extrusion_dir():
    root_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(root_directory, EXTRUSION_DIRECTORY))

def get_extrusion_path(extrusion_name):
    if extrusion_name.endswith('.json'):
        filename = os.path.basename(extrusion_name)
    elif extrusion_name in EXTRUSION_FILENAMES:
        filename = EXTRUSION_FILENAMES[extrusion_name]
    else:
        filename = '{}.json'.format(extrusion_name)
    root_directory = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(root_directory, EXTRUSION_DIRECTORY, filename))

def enumerate_problems():
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), EXTRUSION_DIRECTORY))
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json'):
            yield extrusion_name_from_path(filename)

def extrusion_name_from_path(extrusion_path):
    extrusion_name, _ = os.path.splitext(os.path.basename(extrusion_path))
    return extrusion_name

def load_extrusion(extrusion_path, verbose=False):
    if verbose:
        extrusion_name = extrusion_name_from_path(extrusion_path)
        print('Name: {}'.format(extrusion_name))
        print('Path: {}'.format(extrusion_path))
    with open(extrusion_path, 'r') as f:
        json_data = json.loads(f.read())
    assert json_data['unit'] == 'millimeter'

    element_from_id = OrderedDict((element.id, element.nodes) for element in parse_elements(json_data))
    node_points = parse_node_points(json_data)
    min_z = np.min(node_points, axis=0)[2]
    #print('Min z: {}'.format(min_z))
    node_points = [np.array([0, 0, -min_z]) + point for point in node_points]
    ground_nodes = parse_ground_nodes(json_data)
    if verbose:
        print('Assembly: {} | Model: {} | Unit: {}'.format(
            json_data['assembly_type'], json_data['model_type'], json_data['unit'])) # extrusion, spatial_frame, millimeter
        print('Nodes: {} | Ground: {} | Elements: {}'.format(
            len(node_points), len(ground_nodes), len(element_from_id)))
    # TODO: named tuple
    return element_from_id, node_points, ground_nodes


def parse_point(json_point, scale=DEFAULT_SCALE):
    return scale * np.array([json_point['X'], json_point['Y'], json_point['Z']])


def parse_transform(json_transform, **kwargs):
    transform = np.eye(4)
    transform[:3, 3] = parse_point(json_transform['Origin'], **kwargs) # Normal
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

def json_from_point(point):
    return dict(zip(['X', 'Y', 'Z'], point))

def affine_extrusion(extrusion_path, tform):
    with open(extrusion_path, 'r') as f:
        data = json.loads(f.read())
    new_data = {}
    for key, value in data.items():
        if key == 'base_frame_in_rob_base':
            origin = parse_point(value['Origin'], scale=1)
            new_origin = tform_point(tform, origin)
            rob_from_base = pose_from_tform(parse_transform(value, scale=1))
            new_pose = tform_from_pose(multiply(tform, rob_from_base))
            x, y, z = new_pose[:3,3]
            new_data[key] = {
                'Origin': json_from_point(new_origin),
                "OriginX": x,
                "OriginY": y,
                "OriginZ": z,
                "XAxis": json_from_point(new_pose[0,:3]),
                "YAxis": json_from_point(new_pose[1,:3]),
                "ZAxis": json_from_point(new_pose[2,:3]),
                "IsValid": value['IsValid'],
            }
        elif key == 'node_list':
            new_data[key] = []
            for node_data in value:
                new_node_data = {}
                for node_key in node_data:
                    if node_key == 'point':
                        point = parse_point(node_data[node_key], scale=1)
                        new_point = tform_point(tform, point)
                        new_node_data[node_key] = json_from_point(new_point)
                    else:
                        new_node_data[node_key] = node_data[node_key]
                new_data[key].append(new_node_data)
        else:
            new_data[key] = value
    return new_data

##################################################

def create_elements_bodies(node_points, elements, color=apply_alpha(RED, alpha=1)):
    # TODO: just shrink the structure to prevent worrying about collisions at end-points
    # TODO: could scale the whole environment
    # radius = 1e-6 # 5e-5 | 1e-4
    # radius = 1e-6
    radius = RADIUS
    # TODO: seems to be a min radius

    shrink = SHRINK # 0. | 0.002 | 0.005
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
