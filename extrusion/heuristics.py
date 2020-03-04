import heapq
import random
import numpy as np

from collections import namedtuple

from extrusion.equilibrium import compute_all_reactions, compute_node_reactions
from extrusion.parsing import load_extrusion
from extrusion.utils import get_extructed_ids, downselect_elements, compute_z_distance, TOOL_LINK, get_undirected, \
    reverse_element, get_midpoint, nodes_from_elements, compute_printed_nodes, compute_transit_distance, \
    compute_sequence_distance, compute_element_distance
from extrusion.stiffness import create_stiffness_checker, force_from_reaction, torque_from_reaction, plan_stiffness
from extrusion.tsp import compute_component_mst, solve_tsp, compute_layer_from_directed
from pddlstream.utils import adjacent_from_edges, hash_or_id, get_connected_components, outgoing_from_edges
from pybullet_tools.utils import get_distance, INF, get_joint_positions, get_movable_joints, get_link_pose, \
    link_from_name, BodySaver, set_joint_positions, point_from_pose, get_pitch, set_configuration

DISTANCE_HEURISTICS = [
    'z',
    'dijkstra',
    #'online-dijkstra',
    'plan-stiffness', # TODO: recategorize
]
COST_HEURISTICS = [
    'distance',
    'layered-distance',
    #'mst',
    #'tsp',
    #'online-tsp',
    #'components',
]
TOPOLOGICAL_HEURISTICS = [
    'length',
    'degree',
]
STIFFNESS_HEURISTICS = [
    'stiffness',  # Performs poorly with respect to stiffness
    'fixed-stiffness',
    'relative-stiffness',
    'load',
    'forces',
    'fixed-forces',
]
HEURISTICS = ['none'] + DISTANCE_HEURISTICS + COST_HEURISTICS

##################################################

Node = namedtuple('Node', ['edge', 'vertex', 'cost'])

def dijkstra(source_vertices, successor_fn, cost_fn=lambda v1, v2: 1):
    # TODO: could unify with motion_planners
    node_from_vertex = {}
    queue = []
    for vertex in source_vertices:
        cost = 0
        node_from_vertex[vertex] = Node(None, None, cost)
        heapq.heappush(queue, (cost, vertex))
    while queue:
        cost1, vertex1 = heapq.heappop(queue)
        if node_from_vertex[vertex1].cost < cost1:
            continue
        for vertex2 in successor_fn(vertex1):
            edge = (vertex1, vertex2)
            cost2 = cost1 + cost_fn(*edge)
            if (vertex2 not in node_from_vertex) or (cost2 < node_from_vertex[vertex2].cost):
                node_from_vertex[vertex2] = Node(edge, vertex1, cost2)
                heapq.heappush(queue, (cost2, vertex2))
    return node_from_vertex

##################################################

def compute_distance_from_node(elements, node_points, ground_nodes):
    #incoming_supporters, _ = neighbors_from_orders(get_supported_orders(
    #    element_from_id.values(), node_points))
    nodes = nodes_from_elements(elements)
    neighbors = adjacent_from_edges(elements)
    edge_costs = {edge: get_distance(node_points[edge[0]], node_points[edge[1]])
                  for edge in elements}
    #edge_costs = {edge: np.abs(node_points[edge[1]] - node_points[edge[0]])[2]
    #              for edge in elements}
    edge_costs.update({edge[::-1]: distance for edge, distance in edge_costs.items()})
    successor_fn = lambda v: neighbors[v]
    cost_fn = lambda v1, v2: edge_costs[v1, v2]
    return dijkstra(ground_nodes & nodes, successor_fn, cost_fn)

def compute_layer_from_vertex(elements, node_points, ground_nodes):
    node_from_vertex = compute_distance_from_node(elements, node_points, ground_nodes)
    partial_orders = {(node.vertex, vertex) for vertex, node in node_from_vertex.items() if node.vertex is not None}
    successors_from_vertex = outgoing_from_edges(partial_orders)
    successor_fn = lambda v: successors_from_vertex[v]
    return {vertex: node.cost for vertex, node in dijkstra(ground_nodes, successor_fn).items()}

def compute_layer_from_element(elements, node_points, ground_nodes):
    # TODO: rigid partial orders derived from this
    layer_from_vertex = compute_layer_from_vertex(elements, node_points, ground_nodes)
    layer_from_edge = {e: min(layer_from_vertex[v] for v in e) for e in elements}
    return layer_from_edge

def downsample_structure(elements, node_points, ground_nodes, num=None):
    if num is None:
        return elements
    cost_from_nodes = compute_distance_from_node(elements, node_points, ground_nodes)
    selected_nodes = sorted(cost_from_nodes, key=lambda n: cost_from_nodes[0])[:num] # TODO: bug
    return downselect_elements(elements, selected_nodes)

##################################################

def score_stiffness(extrusion_path, element_from_id, elements, checker=None):
    if not elements:
        return 0
    if checker is None:
        checker = create_stiffness_checker(extrusion_path)
    # TODO: analyze fixities projections in the xy plane

    # Lower is better
    extruded_ids = get_extructed_ids(element_from_id, elements)
    checker.solve(exist_element_ids=extruded_ids, if_cond_num=True)
    success, nodal_displacement, fixities_reaction, _ = checker.get_solved_results()
    if not success:
        return INF
    #operation = np.max
    operation = np.sum # equivalently average
    # TODO: LInf or L1 norm applied on forces
    # TODO: looking for a colored path through the space

    # trans unit: meter, rot unit: rad
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    max_trans, max_rot, _, _ = checker.get_max_nodal_deformation()
    relative_trans = max_trans / trans_tol # lower is better
    relative_rot = max_rot / rot_tol # lower is better
    # More quickly approximate deformations by modifying the matrix operations incrementally

    reaction_forces = np.array([force_from_reaction(d) for d in fixities_reaction.values()])
    reaction_moments = np.array([torque_from_reaction(d) for d in fixities_reaction.values()])
    heuristic = 'compliance'
    scores = {
        # Yijiang was surprised that fixities_translation worked
        'fixities_translation': np.linalg.norm(reaction_forces, axis=1), # Bad when forward
        'fixities_rotation': np.linalg.norm(reaction_moments, axis=1), # Bad when forward
        'nodal_translation': np.linalg.norm(list(nodal_displacement.values()), axis=1),
        'compliance': [checker.get_compliance()],
        'deformation': [relative_trans, relative_rot],
    }
    # TODO: remove pairs of elements
    # TODO: clustering
    return operation(scores[heuristic])
    #return relative_trans
    #return max(relative_trans, relative_rot)
    #return relative_trans + relative_rot # arithmetic mean
    #return relative_trans * relative_rot # geometric mean
    #return 2*relative_trans * relative_rot / (relative_trans + relative_rot) # harmonic mean

##################################################

def get_tool_position(robot):
    tool_link = link_from_name(robot, TOOL_LINK)
    #initial_conf = get_joint_positions(robot, joints)
    initial_pose = get_link_pose(robot, tool_link)
    return point_from_pose(initial_pose)

def get_heuristic_fn(robot, extrusion_path, heuristic, forward, checker=None):
    initial_point = get_tool_position(robot)

    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    all_elements = frozenset(element_from_id.values())
    sign = +1 if forward else -1

    distance_from_node = compute_distance_from_node(all_elements, node_points, ground_nodes)
    layer_from_edge = compute_layer_from_element(all_elements, node_points, ground_nodes)
    _, layer_from_directed = compute_layer_from_directed(all_elements, node_points, ground_nodes)

    tsp_cache = {} # Technically influenced by the current position as well
    plan = None
    if heuristic == 'fixed-tsp':
        plan, _ = solve_tsp(all_elements, ground_nodes, node_points, set(), initial_point, initial_point, visualize=False)
        #tsp_cache[all_elements] = plan
    elif heuristic == 'plan-stiffness':
        plan = plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, all_elements,
                              initial_position=initial_point, checker=checker, max_backtrack=INF)
    order = None
    if plan is not None:
        order = {get_undirected(all_elements, directed): i for i, directed in enumerate(plan)}

    stiffness_cache = {}
    if heuristic in ('fixed-stiffness', 'relative-stiffness'):
        stiffness_cache.update({element: score_stiffness(extrusion_path, element_from_id, all_elements - {element},
                                                         checker=checker) for element in all_elements})

    last_plan = []
    reaction_cache = {}
    distance_cache = {}
    ee_cache = {}
    # TODO: round values for more tie-breaking opportunities
    # TODO: compute for all all_elements up front, sort, and bucket for the score (more general than rounding)

    def fn(printed, directed, position, conf):
        # Queue minimizes the statistic
        element = get_undirected(all_elements, directed)
        n1, n2 = directed

        # forward adds, backward removes
        structure = printed | {element} if forward else printed - {element}
        structure_ids = get_extructed_ids(element_from_id, structure)

        normalizer = 1
        #normalizer = len(structure)
        #normalizer = compute_element_distance(node_points, all_elements)

        reduce_op = sum # sum | max | average
        reaction_fn = force_from_reaction  # force_from_reaction | torque_from_reaction

        first_node, second_node = directed if forward else reverse_element(directed)
        layer = sign * layer_from_edge[element]
        #layer = sign * layer_from_directed.get(directed, INF)

        tool_point = position
        tool_distance = 0.
        if heuristic in COST_HEURISTICS:
            if conf is not None:
                if hash_or_id(conf) not in ee_cache:
                    with BodySaver(robot):
                        set_configuration(robot, conf)
                        ee_cache[hash_or_id(conf)] = get_tool_position(robot)
                tool_point = ee_cache[hash_or_id(conf)]
            tool_distance = get_distance(tool_point, node_points[first_node])

        # TODO: weighted average to balance cost and bias
        if heuristic == 'none':
            return 0
        if heuristic == 'random':
            return random.random()
        elif heuristic == 'degree':
            # TODO: other graph statistics
            #printed_nodes = {n for e in printed for n in e}
            #node = n1 if n2 in printed_nodes else n2
            #if node in ground_nodes:
            #    return 0
            raise NotImplementedError()
        elif heuristic == 'length':
            # Equivalent to mass if uniform density
            return get_distance(node_points[n2], node_points[n1])
        elif heuristic == 'distance':
            return tool_distance
        elif heuristic == 'layered-distance':
            return (layer, tool_distance)
        # elif heuristic == 'components':
        #     # Ground nodes intentionally omitted
        #     # TODO: this is broken
        #     remaining = all_elements - printed if forward else printed - {element}
        #     vertices = nodes_from_elements(remaining)
        #     components = get_connected_components(vertices, remaining)
        #     #print('Components: {} | Distance: {:.3f}'.format(len(components), tool_distance))
        #     return (len(components), tool_distance)
        elif heuristic == 'fixed-tsp':
            # TODO: layer_from_edge[element]
            # TODO: score based on current distance from the plan in the tour
            # TODO: factor in the distance to the next element in a more effective way
            if order is None:
                return (INF, tool_distance)
            return (sign*order[element], tool_distance) # Chooses least expensive direction
        elif heuristic == 'tsp':
            if printed not in tsp_cache: # not last_plan and
                # TODO: seed with the previous solution
                #remaining = all_elements - printed if forward else printed
                #assert element in remaining
                #printed_nodes = compute_printed_nodes(ground_nodes, printed) if forward else ground_nodes
                tsp_cache[printed] = solve_tsp(all_elements, ground_nodes, node_points, printed, tool_point, initial_point,
                                               bidirectional=True, layers=True, max_time=30, visualize=False, verbose=True)
                #print(tsp_cache[printed])
                if not last_plan:
                    last_plan[:] = tsp_cache[printed][0]
            plan, cost = tsp_cache[printed]
            #plan = last_plan[len(printed):]
            if plan is None:
                #return tool_distance
                return (layer, INF)
            transit = compute_transit_distance(node_points, plan, start=tool_point, end=initial_point)
            assert forward
            first = plan[0] == directed
            #return not first # No info if the best isn't possible
            index = None
            for i, directed2 in enumerate(plan):
                undirected2 = get_undirected(all_elements, directed2)
                if element == undirected2:
                    assert index is None
                    index = i
            assert index is not None
            # Could also break ties by other in the plan
            # Two plans might have the same cost but the swap might be detrimental
            new_plan = [directed] + plan[:index] + plan[index+1:]
            assert len(plan) == len(new_plan)
            new_transit = compute_transit_distance(node_points, new_plan, start=tool_point, end=initial_point)
            #print(layer, cost, transit + compute_element_distance(node_points, plan),
            #      new_transit + compute_element_distance(node_points, plan))
            #return new_transit
            return (layer, not first, new_transit) # Layer important otherwise it shortcuts
        elif heuristic == 'online-tsp':
            if forward:
                _, tsp_distance = solve_tsp(all_elements-structure, ground_nodes, node_points, printed,
                                            node_points[second_node], initial_point, visualize=False)
            else:
                _, tsp_distance = solve_tsp(structure, ground_nodes, node_points, printed, initial_point,
                                            node_points[second_node], visualize=False)
            total = tool_distance + tsp_distance
            return total
        # elif heuristic == 'mst':
        #     # TODO: this is broken
        #     mst_distance = compute_component_mst(node_points, ground_nodes, remaining,
        #                                          initial_position=node_points[second_node])
        #     return tool_distance + mst_distance
        elif heuristic == 'x':
            return sign * get_midpoint(node_points, element)[0]
        elif heuristic == 'z':
            return sign * compute_z_distance(node_points, element)
        elif heuristic == 'pitch':
            #delta = node_points[second_node] - node_points[first_node]
            delta = node_points[n2] - node_points[n1]
            return get_pitch(delta)
        elif heuristic == 'dijkstra': # offline
            # TODO: sum of all element path distances
            return sign*np.average([distance_from_node[node].cost for node in element]) # min, max, average
        elif heuristic == 'online-dijkstra':
            if printed not in distance_cache:
                distance_cache[printed] = compute_distance_from_node(printed, node_points, ground_nodes)
            return sign*min(distance_cache[printed][node].cost
                            if node in distance_cache[printed] else INF
                            for node in element)
        elif heuristic == 'plan-stiffness':
            if order is None:
                return None
            return (sign*order[element], directed not in order)
        elif heuristic == 'load':
            nodal_loads = checker.get_nodal_loads(existing_ids=structure_ids, dof_flattened=False) # get_self_weight_loads
            return reduce_op(np.linalg.norm(force_from_reaction(reaction)) for reaction in nodal_loads.values())
        elif heuristic == 'fixed-forces':
            #printed = all_elements # disable to use most up-to-date
            # TODO: relative to the load introduced
            if printed not in reaction_cache:
                reaction_cache[printed] = compute_all_reactions(extrusion_path, all_elements, checker=checker)
            force = reduce_op(np.linalg.norm(reaction_fn(reaction)) for reaction in reaction_cache[printed].reactions[element])
            return force / normalizer
        elif heuristic == 'forces':
            reactions_from_nodes = compute_node_reactions(extrusion_path, structure, checker=checker)
            #torque = sum(np.linalg.norm(np.sum([torque_from_reaction(reaction) for reaction in reactions], axis=0))
            #            for reactions in reactions_from_nodes.values())
            #return torque / normalizer
            total = reduce_op(np.linalg.norm(reaction_fn(reaction)) for reactions in reactions_from_nodes.values()
                            for reaction in reactions)
            return total / normalizer
            #return max(sum(np.linalg.norm(reaction_fn(reaction)) for reaction in reactions)
            #               for reactions in reactions_from_nodes.values())
        elif heuristic == 'stiffness':
            # TODO: add different variations
            # TODO: normalize by initial stiffness, length, or degree
            # Most unstable or least unstable first
            # Gets faster with fewer all_elements
            #old_stiffness = score_stiffness(extrusion_path, element_from_id, printed, checker=checker)
            stiffness = score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
            return stiffness / normalizer
            #return stiffness / old_stiffness
        elif heuristic == 'fixed-stiffness':
            # TODO: invert the sign for regression/progression?
            # TODO: sort FastDownward by the (fixed) action cost
            return stiffness_cache[element] / normalizer
        elif heuristic == 'relative-stiffness':
            stiffness = score_stiffness(extrusion_path, element_from_id, structure, checker=checker) # lower is better
            if normalizer == 0:
                return 0
            return stiffness / normalizer
            #return stiffness / stiffness_cache[element]
        raise ValueError(heuristic)
    return fn
