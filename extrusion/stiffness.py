import heapq
import os
import random
import time
import numpy as np
import math

from collections import namedtuple, defaultdict
from itertools import combinations, product

from pyconmech import StiffnessChecker

from extrusion.utils import get_extructed_ids, compute_sequence_distance, get_distance, compute_printable_directed, get_undirected, \
    get_pairs, compute_z_distance
from pddlstream.utils import get_connected_components
from pybullet_tools.utils import HideOutput, INF, elapsed_time, wait_for_user, BLUE, RED

TRANS_TOL = 0.0015
ROT_TOL = INF # 5 * np.pi / 180
INITIAL_NODE = None

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

SCALE = 1e3

def solve_tsp(elements, node_points, initial_point, max_time=5, verbose=False):
    # https://developers.google.com/optimization/routing/tsp
    # https://developers.google.com/optimization/reference/constraint_solver/routing/RoutingModel
    # AddDisjunction
    # TODO: pick up and delivery
    # TODO: time window for skipping elements
    # TODO: Minimum Spanning Tree (MST) bias
    # TODO: time window constraint to ensure connected
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    from extrusion.visualization import draw_ordered
    assert initial_point is not None
    start_time = time.time()

    #nodes = set(range(len(node_points)))
    nodes = {node for element in elements for node in element}
    point_from_node = dict(enumerate(node_points))
    point_from_node[INITIAL_NODE] = initial_point
    # for element in elements:
    #     point_from_node[element] = get_midpoint(node_points, element)
    # TODO: include midpoints
    node_from_index = [INITIAL_NODE] + sorted(nodes) # + sorted(elements)
    index_from_node = dict(map(reversed, enumerate(node_from_index)))

    num_vehicles, depot = 1, 0
    manager = pywrapcp.RoutingIndexManager(len(node_from_index), num_vehicles, depot)
    solver = pywrapcp.RoutingModel(manager)
    #print(solver.GetAllDimensionNames())
    #print(solver.ComputeLowerBound())

    distance_from_node = {}
    for n1, n2 in product(point_from_node, repeat=2):
        i1, i2 = index_from_node[n1], index_from_node[n2]
        p1, p2 = point_from_node[n1], point_from_node[n2]
        distance_from_node[i1, i2] = int(math.ceil(SCALE*get_distance(p1, p2)))

    # def time_callback(from_index, to_index):
    #     """Returns the travel time between the two nodes."""
    #     # Convert from routing variable Index to time matrix NodeIndex.
    #     from_node = manager.IndexToNode(from_index)
    #     to_node = manager.IndexToNode(to_index)
    #     return 1
    #     #return data['time_matrix'][from_node][to_node]
    #
    # transit_callback_index = solver.RegisterTransitCallback(time_callback)
    # step = 'Time'
    # solver.AddDimension(
    #     transit_callback_index,
    #     30,  # allow waiting time
    #     30,  # maximum time per vehicle
    #     False,  # Don't force start cumul to zero.
    #     step)
    #
    # time_dimension = solver.GetDimensionOrDie(step)
    # for node, index in index_from_node.items():
    #     time_dimension.CumulVar(manager.NodeToIndex(index))

    transit_callback = solver.RegisterTransitCallback(
        lambda i1, i2: distance_from_node[manager.IndexToNode(i1), manager.IndexToNode(i2)])
    solver.SetArcCostEvaluatorOfAllVehicles(transit_callback)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.solution_limit = 1
    search_parameters.time_limit.seconds = int(max_time)
    search_parameters.log_search = verbose
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC) # AUTOMATIC | PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC) # AUTOMATIC | GUIDED_LOCAL_SEARCH
    assignment = solver.SolveWithParameters(search_parameters)
    #initial_solution = solver.ReadAssignmentFromRoutes(data['initial_routes'], True)

    #print('Status: {}'.format(solver.status()))
    if not assignment:
        print('Failure! Duration: {:.3f}s'.format(elapsed_time(start_time)))
        return None
    print('Success! Objective: {:.3f}, Duration: {:.3f}s'.format(
        assignment.ObjectiveValue() / SCALE, elapsed_time(start_time)))
    index = solver.Start(0)
    order = []
    while not solver.IsEnd(index):
        order.append(node_from_index[manager.IndexToNode(index)])
        #previous_index = index
        index = assignment.Value(solver.NextVar(index))
        #route_distance += solver.GetArcCostForVehicle(previous_index, index, 0)
    order.append(node_from_index[manager.IndexToNode(index)])

    print(order)
    edges = list(zip(order[:-1], order[1:]))
    draw_ordered(edges, point_from_node)
    wait_for_user()

    return order

##################################################

def compute_spanning_tree(edge_weights):
    tree = set()
    connected = set()
    if not edge_weights:
        return tree
    adjacent = defaultdict(set)
    for e in edge_weights:
        for v in e:
            adjacent[v].add(e)
    root, _ = random.choice(list(edge_weights))
    queue = []
    for edge in adjacent[root]:
        heapq.heappush(queue, (edge_weights[edge], edge))
    while queue:
        weight, edge = heapq.heappop(queue)
        if set(edge) <= connected:
            continue
        vertex = edge[0 if edge[0] not in connected else 1]
        tree.add(edge)
        connected.update(edge)
        for edge in adjacent[vertex]:
            heapq.heappush(queue, (edge_weights[edge], edge))
    return tree


def compute_euclidean_tree(node_points, ground_nodes, elements, initial_position=None):
    # remove printed elements from the tree
    start_time = time.time()
    point_from_vertex = dict(enumerate(node_points))
    edges = set(elements)
    for element in elements:
        n1, n2 = element
        path = [n1]
        for t in np.linspace(0, 1, num=3, endpoint=True)[1:-1]:
            n3 = len(point_from_vertex)
            point_from_vertex[n3] = t*node_points[n1] + (1-t)*node_points[n2]
            path.append(n3)
        path.append(n2)
        edges.update(get_pairs(path))

    if initial_position is not None:
        point_from_vertex[INITIAL_NODE] = initial_position
        edges.update(set(combinations(ground_nodes | {INITIAL_NODE}, r=2)))
        #edges.update({(INITIAL_NODE, n) for n in ground_nodes})
    edge_weights = {(n1, n2): get_distance(point_from_vertex[n1], point_from_vertex[n2]) for n1, n2 in edges}
    components = get_connected_components(point_from_vertex, edge_weights)
    tree = compute_spanning_tree(edge_weights)
    from extrusion.visualization import draw_model
    draw_model(tree, point_from_vertex, ground_nodes, color=BLUE)
    draw_model(edges - tree, point_from_vertex, ground_nodes, color=RED)

    weight = sum(edge_weights[e] for e in tree)
    print(len(components), len(point_from_vertex), len(tree), weight, elapsed_time(start_time))
    wait_for_user()
    return weight

def compute_component_mst(node_points, ground_nodes, unprinted, initial_position=None):
    start_time = time.time()
    point_from_vertex = dict(enumerate(node_points))
    vertices = {v for e in unprinted for v in e}
    components = get_connected_components(vertices, unprinted)

    entry_nodes = set(ground_nodes)
    if initial_position is not None:
        point_from_vertex[INITIAL_NODE] = initial_position
        components.append([INITIAL_NODE])
        entry_nodes.add(INITIAL_NODE)

    edge_weights = {}
    for c1, c2 in combinations(range(len(components)), r=2):
        # TODO: directed edges from all points to entry nodes
        nodes1 = set(components[c1])
        nodes2 = set(components[c2])
        if (c1 == [INITIAL_NODE]) or (c2 == [INITIAL_NODE]):
            nodes1 &= entry_nodes
            nodes2 &= entry_nodes
        edge_weights[c1, c2] = min(get_distance(point_from_vertex[n1], point_from_vertex[n2])
                                   for n1, n2 in product(nodes1, nodes2))

    tree = compute_spanning_tree(edge_weights)
    weight = sum(edge_weights[e] for e in tree)
    print('Elements: {} | Components: {} | Tree: {} | Weight: {}: Time: {:.3f} '.format(
        len(unprinted), len(components), tree, weight, elapsed_time(start_time)))
    return weight

##################################################

def plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, elements,
                   initial_position=None, checker=None, max_time=INF, max_backtrack=0):
    #assert compute_component_mst(node_points, ground_nodes, elements, initial_position)
    #return compute_euclidean_tree(node_points, ground_nodes, elements, initial_position)
    #return solve_tsp(elements, node_points, initial_position)
    start_time = time.time()
    if checker is None:
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
        if not test_stiffness(extrusion_path, element_from_id, printed, checker=checker, verbose=False):
            continue
        if printed == remaining_elements:
            #from extrusion.visualization import draw_ordered
            distance = compute_sequence_distance(node_points, sequence)
            print('Success! Elements: {}, Distance: {:.2f}m, Time: {:.3f}sec'.format(
                len(sequence), distance, elapsed_time(start_time)))
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
            #distance = get_distance(position, node_points[node1]) if position is not None else None
            #distance = compute_sequence_distance(node_points, new_sequence)
            #bias = None
            bias = compute_z_distance(node_points, element)
            #bias = distance
            #bias = random.random()
            #bias = heuristic_fn(printed, element, conf=None) # TODO: experiment with other biases
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, new_printed, node_points[node2], new_sequence))
    print('Failed to find stiffness plan! Elements: {}, Min remaining {}, Time: {:.3f}sec'.format(
        len(remaining_elements), min_remaining, elapsed_time(start_time)))
    return None
