import heapq
import os
import random
import time
from collections import namedtuple

import numpy as np
from pyconmech import StiffnessChecker

from extrusion.utils import get_extructed_ids, compute_printable_elements, compute_z_distance, \
    compute_sequence_distance, compute_printed_nodes, is_printable
from pybullet_tools.utils import HideOutput, INF, elapsed_time, randomize

TRANS_TOL = 0.0015
ROT_TOL = INF # 5 * np.pi / 180

Deformation = namedtuple('Deformation', ['success', 'displacements', 'fixities', 'reactions']) # TODO: get_max_nodal_deformation
Displacement = namedtuple('Displacement', ['dx', 'dy', 'dz', 'theta_x', 'theta_y', 'theta_z'])
Reaction = namedtuple('Reaction', ['fx', 'fy', 'fz', 'mx', 'my', 'mz'])


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


def force_from_reaction(reaction):
    return reaction[:3]


def torque_from_reaction(reaction):
    return reaction[3:]


def evaluate_stiffness(extrusion_path, element_from_id, elements, checker=None, verbose=True):
    # TODO: check each connected component individually
    if not elements:
        return Deformation(True, {}, {}, {})
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
    assert is_stiff == success
    displacements = {i: Displacement(*d) for i, d in nodal_displacement.items()}
    fixities = {i: Reaction(*d) for i, d in fixities_reaction.items()}
    reactions = {i: (Reaction(*d[0]), Reaction(*d[1])) for i, d in element_reaction.items()}

    #print("nodal displacement (m/rad):\n{0}".format(nodal_displacement)) # nodes x 7
    # TODO: investigate if nodal displacement can be used to select an ordering
    #print("fixities reaction (kN, kN-m):\n{0}".format(fixities_reaction)) # ground x 7
    #print("element reaction (kN, kN-m):\n{0}".format(element_reaction)) # elements x 13
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    max_trans, max_rot, max_trans_vid, max_rot_vid = checker.get_max_nodal_deformation()
    # The inverse of stiffness is flexibility or compliance

    translation = np.max(np.linalg.norm([d[:3] for d in displacements.values()], ord=2, axis=1))
    rotation = np.max(np.linalg.norm([d[3:] for d in displacements.values()], ord=2, axis=1))
    is_stiff &= (translation <= trans_tol) and (rotation <= rot_tol)

    if verbose:
        print('Stiff: {} | Compliance: {:.5f}'.format(is_stiff, checker.get_compliance()))
        print('Max translation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_trans, trans_tol, max_trans / trans_tol, max_trans_vid))
        print('Max rotation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_rot, rot_tol, max_rot / rot_tol, max_rot_vid))
    #disc = 10
    #time_step = 1.0
    #orig_beam_shape = checker.get_original_shape(disc=disc, draw_full_shape=False)
    #beam_disp = checker.get_deformed_shape(exagg_ratio=1.0, disc=disc)
    return Deformation(is_stiff, displacements, fixities, reactions)


def test_stiffness(extrusion_path, element_from_id, elements, **kwargs):
    return evaluate_stiffness(extrusion_path, element_from_id, elements, **kwargs).success

##################################################

# https://developers.google.com/optimization/routing/tsp

def plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, elements,
                   checker=None, max_time=INF, max_backtrack=0):
    # TODO: Minimum Spanning Tree (MST) bias
    start_time = time.time()
    if checker is None:
        checker = create_stiffness_checker(extrusion_path)
    remaining_elements = frozenset(elements)
    min_remaining = len(remaining_elements)
    queue = [(None, frozenset(), None, [])]
    while queue and (elapsed_time(start_time) < max_time):
        _, printed, position, sequence = heapq.heappop(queue)
        num_remaining = len(remaining_elements) - len(printed)
        backtrack = num_remaining - min_remaining
        if max_backtrack < backtrack:
            break # continue
        if not test_stiffness(extrusion_path, element_from_id, printed, checker=checker, verbose=False):
            continue
        if printed == remaining_elements:
            distance = compute_sequence_distance(node_points, sequence)
            print('Success! Elements: {}, Distance: {:.2f}m, Time: {:.3f}sec'.format(
                len(sequence), distance, elapsed_time(start_time)))
            return sequence
        nodes = compute_printed_nodes(ground_nodes, printed)
        #for element in randomize(compute_printable_elements(remaining_elements, ground_nodes, printed)):
        for element in set(remaining_elements) - printed:
            for reverse in [True, False]:
                directed = element[::-1] if reverse else element
                node1, node2 = directed
                if node1 in nodes:
                    new_printed = printed | {element}
                    new_sequence = sequence + [element]
                    num_remaining = len(remaining_elements) - len(new_printed)
                    min_remaining = min(min_remaining, num_remaining)
                    distance = compute_sequence_distance(node_points, new_sequence)
                    #bias = None
                    bias = compute_z_distance(node_points, element)
                    #bias = heuristic_fn(printed, element, conf=None) # TODO: experiment with other biases
                    priority = (num_remaining, bias, random.random())
                    heapq.heappush(queue, (priority, new_printed, node_points[node2], new_sequence))
    print('Failed to find stiffness plan! Elements: {}, Min remaining {}, Time: {:.3f}sec'.format(
        len(remaining_elements), min_remaining, elapsed_time(start_time)))
    return None
