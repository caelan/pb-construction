import heapq
import math
import random
import time
from collections import defaultdict
from itertools import product, combinations

import numpy as np

from extrusion.utils import get_pairs
from pddlstream.utils import get_connected_components
from pybullet_tools.utils import get_distance, elapsed_time, BLACK, wait_for_user, BLUE, RED

INITIAL_NODE = None
SCALE = 1e3
PENALTY = 1e3

##################################################

def solve_tsp(elements, ground_nodes, node_points, initial_point, max_time=30, verbose=False):
    # https://developers.google.com/optimization/routing/tsp
    # https://developers.google.com/optimization/reference/constraint_solver/routing/RoutingModel
    # AddDisjunction
    # TODO: pick up and delivery
    # TODO: time window for skipping elements
    # TODO: Minimum Spanning Tree (MST) bias
    # TODO: time window constraint to ensure connected
    # TODO: reuse by simply swapping out the first vertex
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    from extrusion.visualization import draw_ordered, draw_model
    assert initial_point is not None
    start_time = time.time()
    nodes = {node for element in elements for node in element}
    transit_edges = {pair for pair in product(nodes, repeat=2)}
    transit_edges.update({(INITIAL_NODE, n): 1. for n in ground_nodes}) # initial -> ground
    transit_edges.update({(n, INITIAL_NODE): 1. for n in nodes}) # any -> initial

    point_from_node, frame_edges = embed_graph(elements, node_points, initial_point, num=1)

    edge_weights = {pair: PENALTY for pair in product(point_from_node, repeat=2)}
    for n1, n2 in transit_edges:
        p1, p2 = point_from_node[n1], point_from_node[n2]
        edge_weights[n1, n2] = get_distance(p1, p2)
    edge_weights.update({e: 0. for e in frame_edges}) # frame edges are free

    node_from_index = list(point_from_node)
    index_from_node = dict(map(reversed, enumerate(node_from_index)))
    num_vehicles, depot = 1, 0
    manager = pywrapcp.RoutingIndexManager(len(node_from_index), num_vehicles, depot)
    solver = pywrapcp.RoutingModel(manager)
    #print(solver.GetAllDimensionNames())
    #print(solver.ComputeLowerBound())

    cost_from_index = {}
    for (n1, n2), weight in edge_weights.items():
        i1, i2 = index_from_node[n1], index_from_node[n2]
        cost_from_index[i1, i2] = int(math.ceil(SCALE*weight))

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
        lambda i1, i2: cost_from_index[manager.IndexToNode(i1), manager.IndexToNode(i2)])
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
    print('Success! Vertices: {} | Edges: {} | Objective: {:.3f}, Duration: {:.3f}s'.format(
        len(node_from_index), len(edge_weights), assignment.ObjectiveValue() / SCALE, elapsed_time(start_time)))
    index = solver.Start(0)
    order = []
    while not solver.IsEnd(index):
        order.append(node_from_index[manager.IndexToNode(index)])
        #previous_index = index
        index = assignment.Value(solver.NextVar(index))
        #route_distance += solver.GetArcCostForVehicle(previous_index, index, 0)
    #order.append(node_from_index[manager.IndexToNode(index)])
    start = order.index(INITIAL_NODE)
    order = order[start:] + order[:start] + [INITIAL_NODE]
    print(order)
    # TODO: penalize downwards printing directions by z or phi

    # TODO: visualize by weight
    tour_pairs = get_pairs(order) + get_pairs(list(reversed(order)))
    #draw_model(elements, point_from_node, ground_nodes, color=BLACK)
    draw_model(frame_edges - set(tour_pairs), point_from_node, ground_nodes, color=BLACK)
    draw_ordered(tour_pairs, point_from_node)
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


def embed_graph(elements, node_points, initial_position=None, num=1):
    point_from_vertex = dict(enumerate(node_points))

    if initial_position is not None:
        point_from_vertex[INITIAL_NODE] = initial_position

    frame_edges = set()
    for element in elements:
        n1, n2 = element
        path = [n1]
        for t in np.linspace(0, 1, num=2+num, endpoint=True)[1:-1]:
            n3 = len(point_from_vertex)
            point_from_vertex[n3] = t*node_points[n1] + (1-t)*node_points[n2]
            path.append(n3)
        path.append(n2)
        for directed in get_pairs(path) + get_pairs(list(reversed(path))):
            frame_edges.add(directed)

    return point_from_vertex, frame_edges


def compute_euclidean_tree(node_points, ground_nodes, elements, initial_position=None):
    # remove printed elements from the tree
    # TODO: directed version
    start_time = time.time()
    point_from_vertex, edges = embed_graph(elements, node_points, ground_nodes, initial_position)
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

##################################################

def compute_component_mst(node_points, ground_nodes, unprinted, initial_position=None):
    # Weighted A*
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