import heapq
import os
import random
import time
import numpy as np

from collections import namedtuple
from itertools import combinations, product, combinations_with_replacement

from extrusion.utils import get_extructed_ids, compute_sequence_distance, get_distance, \
    compute_printable_directed, get_undirected, compute_z_distance, reverse_element, get_directions
from pybullet_tools.utils import HideOutput, INF, elapsed_time, randomize, implies, user_input

try:
    import pyconmech
except ImportError as e:
    print('\x1b[6;30;43m' + '{}, Not using conmech'.format(e) + '\x1b[0m')
    USE_CONMECH = False
    user_input("Press Enter to continue...")
else:
    USE_CONMECH = True

TRANS_TOL = 0.0015
ROT_TOL = INF # 5 * np.pi / 180

Deformation = namedtuple('Deformation', ['success', 'displacements', 'fixities', 'reactions']) # TODO: get_max_nodal_deformation
Displacement = namedtuple('Displacement', ['dx', 'dy', 'dz', 'theta_x', 'theta_y', 'theta_z'])
Reaction = namedtuple('Reaction', ['fx', 'fy', 'fz', 'mx', 'my', 'mz'])


def create_stiffness_checker(extrusion_path, verbose=False):
    if not USE_CONMECH:
        return None
    from pyconmech import StiffnessChecker
    # TODO: the stiffness checker likely has a memory leak
    # https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp
    if not os.path.exists(extrusion_path):
        raise FileNotFoundError(extrusion_path)
    with HideOutput():
        #checker = StiffnessChecker(json_file_path=extrusion_path, verbose=verbose)
        checker = StiffnessChecker.from_json(json_file_path=extrusion_path, verbose=verbose)
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

def count_violations(ground_nodes, sequence):
    violations = 0
    printed_nodes = set(ground_nodes)
    for directed in sequence:
        if directed[0] not in printed_nodes:
            #print(directed)
            violations += 1
        printed_nodes.update(directed)
    return violations

def search_neighborhood(extrusion_path, element_from_id, node_points, ground_nodes, checker, sequence,
                        initial_position, stiffness=True, max_candidates=1, max_time=INF):
    start_time = time.time()
    best_sequence = None
    best_cost = compute_sequence_distance(node_points, sequence, start=initial_position, end=initial_position)
    candidates = 1
    for i1, i2 in randomize(combinations_with_replacement(range(len(sequence)), r=2)):
        for directed1 in get_directions(sequence[i1]):
            for directed2 in get_directions(sequence[i2]):
                if implies(best_sequence, (max_candidates <= candidates)) and (max_time <= elapsed_time(start_time)):
                    return best_sequence
                candidates += 1
                if i1 == i2:
                    new_sequence = sequence[:i1] + [directed1] + sequence[i1 + 1:]
                else:
                    new_sequence = sequence[:i1] + [directed1] + sequence[i1 + 1:i2] + [directed2] + sequence[i2 + 1:]
                assert len(new_sequence) == len(sequence)
                if count_violations(ground_nodes, new_sequence):
                    continue
                new_cost = compute_sequence_distance(node_points, new_sequence, start=initial_position,
                                                     end=initial_position)
                if best_cost <= new_cost:
                    continue
                print(best_cost, new_cost)
                return new_sequence
                # TODO: eager version of this also
                # if stiffness and not test_stiffness(extrusion_path, element_from_id, printed, checker=checker, verbose=False):
                #    continue # Unfortunately the full structure is affected
    return best_sequence

def local_search(extrusion_path, element_from_id, node_points, ground_nodes, checker, sequence,
                 initial_position=None, stiffness=True, max_time=INF):
    start_time = time.time()
    sequence = list(sequence)
    #elements = set(element_from_id.values())
    #indices = list(range(len(sequence)))
    #directions = [True, False]

    while elapsed_time(start_time) < max_time:
        new_sequence = search_neighborhood(extrusion_path, element_from_id, node_points, ground_nodes, checker, sequence,
                                           initial_position, stiffness=stiffness, max_time=INF)
        if new_sequence is None:
            break
        sequence = new_sequence

##################################################

def plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, elements,
                   initial_position=None, checker=None, stiffness=True, heuristic='z', max_time=INF, max_backtrack=0):
    start_time = time.time()
    if stiffness and checker is None:
        checker = create_stiffness_checker(extrusion_path)
    remaining_elements = frozenset(elements)
    min_remaining = len(remaining_elements)
    queue = [(None, frozenset(), initial_position, [])]
    while queue and (elapsed_time(start_time) < max_time):
        _, printed, position, sequence = heapq.heappop(queue)
        num_remaining = len(remaining_elements) - len(printed)
        backtrack = num_remaining - min_remaining
        if max_backtrack < backtrack:
            break # continue
        if stiffness and not test_stiffness(extrusion_path, element_from_id, printed, checker=checker, verbose=False):
            continue
        if printed == remaining_elements:
            #from extrusion.visualization import draw_ordered
            distance = compute_sequence_distance(node_points, sequence, start=initial_position, end=initial_position)
            print('Success! Elements: {}, Distance: {:.3f}m, Time: {:.3f}sec'.format(
                len(sequence), distance, elapsed_time(start_time)))
            #local_search(extrusion_path, element_from_id, node_points, ground_nodes, checker, sequence,
            #             initial_position=initial_position, stiffness=stiffness, max_time=INF)
            #draw_ordered(sequence, node_points)
            #wait_for_user()
            return sequence
        for directed in compute_printable_directed(remaining_elements, ground_nodes, printed):
            node1, node2 = directed
            element = get_undirected(elements, directed)
            new_printed = printed | {element}
            new_sequence = sequence + [directed]
            num_remaining = len(remaining_elements) - len(new_printed)
            min_remaining = min(min_remaining, num_remaining)
            # Don't count edge length
            distance = get_distance(position, node_points[node1]) if position is not None else None
            # distance = compute_sequence_distance(node_points, new_sequence)
            if heuristic == 'none':
                bias = None
            elif heuristic == 'random':
                bias = random.random()
            elif heuristic == 'z':
                bias = compute_z_distance(node_points, element)
            elif heuristic == 'distance':
                bias = distance
            else:
                raise ValueError(heuristic)
            #bias = heuristic_fn(printed, element, conf=None) # TODO: experiment with other biases
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, new_printed, node_points[node2], new_sequence))
    print('Failed to find stiffness plan! Elements: {}, Min remaining {}, Time: {:.3f}sec'.format(
        len(remaining_elements), min_remaining, elapsed_time(start_time)))
    return None
