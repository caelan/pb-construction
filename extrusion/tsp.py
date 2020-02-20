import heapq
import math
import random
import time
from collections import defaultdict
from itertools import product, combinations

import numpy as np

from extrusion.utils import get_pairs, get_midpoint, SUPPORT_THETA, get_undirected, compute_element_distance, \
    reverse_element, nodes_from_elements, is_start, is_end, get_other_node, compute_transit_distance, compute_printed_nodes
from pddlstream.utils import get_connected_components
from pybullet_tools.utils import get_distance, elapsed_time, BLACK, wait_for_user, BLUE, RED, get_pitch, INF, \
    angle_between, remove_all_debug, GREEN, draw_point
from extrusion.stiffness import plan_stiffness

INITIAL_NODE = 'initial'
FINAL_NODE = 'final'
SCALE = 1e3 # millimeters
INVALID = 1e3 # meters (just make larger than sum of path)

STATUS = """
ROUTING_NOT_SOLVED: Problem not solved yet.
ROUTING_SUCCESS: Problem solved successfully.
ROUTING_FAIL: No solution found to the problem.
ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution.
ROUTING_INVALID: Model, model parameters, or flags are not valid.
""".strip().split('\n')

##################################################

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    vehicle_id = 0
    #for vehicle_id in range(data['num_vehicles']):
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} -> '.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
    plan_output += '{}\n'.format(manager.IndexToNode(index))
    plan_output += 'Distance of the route: {}m\n'.format(route_distance)
    print(plan_output)
    max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))

def parse_solution(solver, manager, key_from_index, solution):
    index = solver.Start(0)
    order = []
    while not solver.IsEnd(index):
        order.append(key_from_index[manager.IndexToNode(index)])
        # previous_index = index
        index = solution.Value(solver.NextVar(index))
        # route_distance += solver.GetArcCostForVehicle(previous_index, index, 0)
    order.append(key_from_index[manager.IndexToNode(index)])
    return order

##################################################

def greedily_plan(all_elements, node_points, ground_nodes, remaining, initial_point):
    from extrusion.heuristics import compute_distance_from_node, compute_layer_from_vertex, compute_z_distance
    level_from_node = compute_layer_from_vertex(all_elements, node_points, ground_nodes)
    cost_from_edge = {}
    if not all(node in level_from_node for node in nodes_from_elements(remaining)):
        return level_from_node, cost_from_edge, None # Disconnected
    for edge in remaining:  # TODO: might be redundant given compute_layer_from_element
        n1, n2 = edge
        if level_from_node[n1] <= level_from_node[n2]:
            cost_from_edge[n1, n2] = level_from_node[n1]
        else:
            cost_from_edge[n2, n1] = level_from_node[n2]
    # sequence = sorted(tree_elements, key=lambda e: cost_from_edge[e])

    point = initial_point
    sequence = []
    remaining_elements = set(cost_from_edge)
    while remaining_elements:
        #key_fn = lambda d: (cost_from_edge[d], random.random())
        #key_fn = lambda d: (cost_from_edge[d], compute_z_distance(node_points, d))
        key_fn = lambda d: (cost_from_edge[d], get_distance(point, node_points[d[0]]))
        directed = min(remaining_elements, key=key_fn)
        remaining_elements.remove(directed)
        sequence.append(directed)
        point = node_points[directed[1]]
    # draw_ordered(sequence, node_points)
    # wait_for_user()
    return level_from_node, cost_from_edge, sequence

def solve_tsp(all_elements, ground_nodes, node_points, printed, initial_point, final_point, layers=True,
              max_time=30, visualize=True, verbose=False):
    # https://developers.google.com/optimization/routing/tsp
    # https://developers.google.com/optimization/reference/constraint_solver/routing/RoutingModel
    # http://www.math.uwaterloo.ca/tsp/concorde.html
    # https://developers.google.com/optimization/reference/python/constraint_solver/pywrapcp
    # AddDisjunction
    # TODO: pick up and delivery
    # TODO: time window for skipping elements
    # TODO: Minimum Spanning Tree (MST) bias
    # TODO: time window constraint to ensure connected
    # TODO: reuse by simply swapping out the first vertex
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    from extrusion.visualization import draw_ordered, draw_model
    start_time = time.time()
    assert initial_point is not None
    remaining = all_elements - printed
    #printed_nodes = compute_printed_nodes(ground_nodes, printed)
    if not remaining:
        cost = get_distance(initial_point, final_point)
        return [], cost
    # TODO: use as a lower bound
    total_distance = compute_element_distance(node_points, remaining)

    # TODO: some of these are invalid still
    level_from_node, cost_from_edge, sequence = greedily_plan(all_elements, node_points, ground_nodes, remaining, initial_point)
    if sequence is None:
        return None, INF
    extrusion_edges = set()
    point_from_vertex = {INITIAL_NODE: initial_point, FINAL_NODE: final_point}
    frame_nodes = nodes_from_elements(remaining)
    min_level = min(level_from_node[n] for n in frame_nodes)
    max_level = max(level_from_node[n] for n in frame_nodes)

    frame_keys = set()
    keys_from_node = defaultdict(set)
    for element in remaining:
        mid = (element, element)
        point_from_vertex[mid] = get_midpoint(node_points, element)
        for node in element:
            key = (element, node)
            point_from_vertex[key] = node_points[node]
            frame_keys.add(key)
            keys_from_node[node].add(key)

        for reverse in [True, False]:
            directed = reverse_element(element) if reverse else element
            node1, node2 = directed
            #delta = node_points[node2] - node_points[node1]
            #pitch = get_pitch(delta)
            #upward = -SUPPORT_THETA <= pitch
            # theta = angle_between(delta, [0, 0, -1])
            # upward = theta < (np.pi / 2 - SUPPORT_THETA)
            #if (directed in tree_elements): # or upward:
            if directed in cost_from_edge:
                # Add edges from anything that is roughly the correct cost
                start = (element, node1)
                end = (element, node2)
                extrusion_edges.update({(start, mid), (mid, end)})

    for node in keys_from_node:
        for edge in product(keys_from_node[node], repeat=2):
            extrusion_edges.add(edge)
    # Key thing is partial order on buckets of elements to adhere to height

    # Connect v2 to v1 if v2 is the same level
    # Traversing an edge might move to a prior level (but at most one)
    transit_edges = set()
    for directed in product(frame_keys, repeat=2):
        key1, key2 = directed
        element1, node1 = key1
        #level1 = min(level_from_node[n] for n in element1)
        level1 = level_from_node[get_other_node(node1, element1)]
        _, node2 = key2
        level2 = level_from_node[node2]
        if level2 in [level1, level1+1]: # TODO: could bucket more coarsely
            transit_edges.add(directed)
    for key in frame_keys:
        _, node = key
        if level_from_node[node] == min_level:
            transit_edges.add((INITIAL_NODE, key))
        if level_from_node[node] in [max_level, max_level-1]:
            transit_edges.add((key, FINAL_NODE))
    # TODO: can also remove restriction that elements are printed in a single direction
    if not layers:
        transit_edges.update(product(frame_keys | {INITIAL_NODE, FINAL_NODE}, repeat=2)) # TODO: apply to greedy as well

    key_from_index = list({k for pair in extrusion_edges | transit_edges for k in pair})
    edge_weights = {pair: INVALID for pair in product(key_from_index, repeat=2)}
    for k1, k2 in transit_edges:
        p1, p2 = point_from_vertex[k1], point_from_vertex[k2]
        edge_weights[k1, k2] = get_distance(p1, p2)
    edge_weights.update({e: 0. for e in extrusion_edges}) # frame edges are free
    edge_weights[FINAL_NODE, INITIAL_NODE] = 0.

    print('Elements: {} | Vertices: {} | Edges: {} | Structure: {:.3f} | Min Level {} | Max Level: {}'.format(
        len(remaining), len(key_from_index), len(edge_weights), total_distance, min_level, max_level))
    index_from_key = dict(map(reversed, enumerate(key_from_index)))
    num_vehicles, depot = 1, index_from_key[INITIAL_NODE]
    manager = pywrapcp.RoutingIndexManager(len(key_from_index), num_vehicles, depot)
                                           #[depot], [depot])
    solver = pywrapcp.RoutingModel(manager)

    cost_from_index = {}
    for (k1, k2), weight in edge_weights.items():
        i1, i2 = index_from_key[k1], index_from_key[k2]
        cost_from_index[i1, i2] = int(math.ceil(SCALE * weight))
    solver.SetArcCostEvaluatorOfAllVehicles(solver.RegisterTransitCallback(
        lambda i1, i2: cost_from_index[manager.IndexToNode(i1), manager.IndexToNode(i2)]))  # from -> to

    # sequence = plan_stiffness(None, None, node_points, ground_nodes, elements,
    #                           initial_position=initial_point, stiffness=False, max_backtrack=INF)

    initial_order = []
    #initial_order = [INITIAL_NODE] # Start and end automatically included
    for directed in sequence:
        node1, node2 = directed
        element = get_undirected(remaining, directed)
        initial_order.extend([
            (element, node1),
            (element, element),
            (element, node2),
        ])
    initial_order.append(FINAL_NODE)
    #initial_order.append(INITIAL_NODE)
    initial_route = [index_from_key[key] for key in initial_order]
    #index = initial_route.index(0)
    #initial_route = initial_route[index:] + initial_route[:index] + [0]

    initial_solution = solver.ReadAssignmentFromRoutes([initial_route], ignore_inactive_indices=True)
    assert initial_solution is not None
    #print_solution(manager, solver, initial_solution)
    #print(solver.GetAllDimensionNames())
    #print(solver.ComputeLowerBound())

    objective = initial_solution.ObjectiveValue() / SCALE
    invalid = int(objective / INVALID)
    order = parse_solution(solver, manager, key_from_index, initial_solution)[:-1]
    ordered_pairs = get_pairs(order)
    cost = sum(edge_weights[pair] for pair in ordered_pairs)
    print('Initial solution | Invalid: {} | Objective: {:.3f} | Cost: {:.3f} | Duration: {:.3f}s'.format(
        invalid, objective, cost, elapsed_time(start_time)))
    if visualize: # and invalid
        remove_all_debug()
        draw_model(printed, node_points, None, color=BLACK)
        draw_point(initial_point, color=BLACK)
        draw_point(final_point, color=GREEN)
        for pair in ordered_pairs:
            if edge_weights[pair] == INVALID:
                for key in pair:
                    draw_point(point_from_vertex[key], color=RED)
        draw_ordered(ordered_pairs, point_from_vertex)
        wait_for_user() # TODO: pause only if viewer
    start_time = time.time()

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.solution_limit = 1
    search_parameters.time_limit.seconds = int(max_time)
    search_parameters.log_search = verbose
    #search_parameters.first_solution_strategy = (
    #    routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC) # AUTOMATIC | PATH_CHEAPEST_ARC
    #search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC) # AUTOMATIC | GUIDED_LOCAL_SEARCH
    #solution = solver.SolveWithParameters(search_parameters)
    solution = solver.SolveFromAssignmentWithParameters(initial_solution, search_parameters)
    #print_solution(manager, solver, solution)

    print('Status: {} | Text: {}'.format(solver.status(), STATUS[solver.status()]))
    if not solution:
        print('Failure! Duration: {:.3f}s'.format(elapsed_time(start_time)))
        return None, INF

    objective = solution.ObjectiveValue() / SCALE
    invalid = int(objective / INVALID)
    order = parse_solution(solver, manager, key_from_index, solution)[:-1]
    ordered_pairs = get_pairs(order) # + [(order[-1], order[0])]
    #cost = compute_element_distance(point_from_vertex, ordered_pairs)
    cost = sum(edge_weights[pair] for pair in ordered_pairs)
    print('Final solution | Invalid: {} | Objective: {:.3f} | Cost: {:.3f} | Duration: {:.3f}s'.format(
        invalid, objective, cost, elapsed_time(start_time)))
    if visualize:
        # TODO: visualize by weight
        remove_all_debug()
        draw_model(printed, node_points, None, color=BLACK)
        draw_point(initial_point, color=BLACK)
        draw_point(final_point, color=GREEN)
        #tour_pairs = ordered_pairs + get_pairs(list(reversed(order)))
        #draw_model(extrusion_edges - set(tour_pairs), point_from_vertex, ground_nodes, color=BLACK)
        draw_ordered(ordered_pairs, point_from_vertex)
        wait_for_user()

    planned = set()
    sequence = []
    for key1, key2 in ordered_pairs[1:-1]:
        element1, node1 = key1
        element2, node2 = key2
        if (element1 == element2) and (element1 not in planned) and (node1 in level_from_node):
            directed = reverse_element(element1) if is_end(node1, element1) else element1
            planned.add(element1)
            sequence.append(directed)
    if remaining != planned:
        return None, INF
    return sequence, cost

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