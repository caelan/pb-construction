import heapq
import random
import numpy as np

from extrusion.equilibrium import compute_all_reactions, compute_node_reactions
from extrusion.parsing import load_extrusion
from extrusion.utils import get_extructed_ids, downselect_elements, compute_z_distance, TOOL_LINK, get_undirected, reverse_element
from extrusion.stiffness import create_stiffness_checker, force_from_reaction, torque_from_reaction, plan_stiffness, \
    compute_component_mst
from pddlstream.utils import adjacent_from_edges, hash_or_id
from pybullet_tools.utils import get_distance, INF, get_joint_positions, get_movable_joints, get_link_pose, \
    link_from_name, BodySaver, set_joint_positions, point_from_pose

DISTANCE_HEURISTICS = [
    'z',
    'dijkstra',
    #'online-dijkstra',
    'plan-stiffness', # TODO: recategorize
]
COST_HEURISTICS = [
    'distance',
    'mst',
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

def compute_distance_from_node(elements, node_points, ground_nodes):
    #incoming_supporters, _ = neighbors_from_orders(get_supported_orders(
    #    element_from_id.values(), node_points))
    neighbors = adjacent_from_edges(elements)
    edge_costs = {edge: get_distance(node_points[edge[0]], node_points[edge[1]])
                  for edge in elements}
    edge_costs.update({edge[::-1]: distance for edge, distance in edge_costs.items()})

    cost_from_node = {}
    queue = []
    cost = 0
    for node in ground_nodes:
        cost_from_node[node] = cost
        heapq.heappush(queue, (cost, node))
    while queue:
        cost1, node1 = heapq.heappop(queue)
        if cost_from_node[node1] < cost1:
            continue
        for node2 in neighbors[node1]:
            cost2 = cost1 + edge_costs[node1, node2]
            if cost2 < cost_from_node.get(node2, INF):
                cost_from_node[node2] = cost2
                heapq.heappush(queue, (cost2, node2))
    return cost_from_node

def downsample_structure(elements, node_points, ground_nodes, num=None):
    if num is None:
        return elements
    cost_from_nodes = compute_distance_from_node(elements, node_points, ground_nodes)
    selected_nodes = sorted(cost_from_nodes, key=lambda n: cost_from_nodes[0])[:num]
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

def get_heuristic_fn(robot, extrusion_path, heuristic, forward, checker=None):
    joints = get_movable_joints(robot)
    tool_link = link_from_name(robot, TOOL_LINK)
    #initial_conf = get_joint_positions(robot, joints)
    initial_pose = get_link_pose(robot, tool_link)

    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    all_elements = frozenset(element_from_id.values())
    distance_from_node = compute_distance_from_node(all_elements, node_points, ground_nodes)
    sign = +1 if forward else -1

    stiffness_order = None
    if heuristic == 'plan-stiffness':
        initial_point = point_from_pose(initial_pose)
        stiffness_plan = plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, all_elements,
                                        initial_position=initial_point, checker=checker, max_backtrack=INF)
        if stiffness_plan is not None:
            stiffness_order = {get_undirected(all_elements, directed): i for i, directed in enumerate(stiffness_plan)}

    stiffness_cache = {}
    if heuristic in ('fixed-stiffness', 'relative-stiffness'):
        stiffness_cache.update({element: score_stiffness(extrusion_path, element_from_id, all_elements - {element},
                                                         checker=checker) for element in all_elements})
    reaction_cache = {}
    distance_cache = {}
    ee_cache = {}
    # TODO: round values for more tie-breaking opportunities
    # TODO: compute for all all_elements up front, sort, and bucket for the score (more general than rounding)

    def fn(printed, directed, conf):
        # Queue minimizes the statistic
        element = get_undirected(all_elements, directed)
        structure = printed | {element} if forward else printed - {element}
        structure_ids = get_extructed_ids(element_from_id, structure)
        # TODO: build away from the robot (x)
        # TODO: order all_elements by angle

        normalizer = 1
        #normalizer = len(structure)
        #normalizer = compute_element_distance(node_points, all_elements)

        reduce_op = sum # sum | max | average
        reaction_fn = force_from_reaction  # force_from_reaction | torque_from_reaction

        tool_distance = 0.
        if heuristic in ['distance', 'mst']:
            assert conf is not None
            if hash_or_id(conf) not in ee_cache:
                with BodySaver(robot):
                    set_joint_positions(robot, joints, conf)
                    ee_cache[hash_or_id(conf)] = get_link_pose(robot, tool_link)
            tool_point = point_from_pose(ee_cache[hash_or_id(conf)])
            first_node, second_node = directed if forward else reverse_element(directed)
            tool_distance = get_distance(tool_point, node_points[first_node])

        if heuristic == 'none':
            return 0
        if heuristic == 'random':
            return random.random()
        elif heuristic == 'degree':
            # TODO: other graph statistics
            #printed_nodes = {n for e in printed for n in e}
            #n1, n2 = element
            #node = n1 if n2 in printed_nodes else n2
            #if node in ground_nodes:
            #    return 0
            raise NotImplementedError()
        elif heuristic == 'length':
            # Equivalent to mass if uniform density
            n1, n2 = element
            return get_distance(node_points[n2], node_points[n1])
        elif heuristic == 'distance':
            return tool_distance
        elif heuristic == 'mst':
            remaining_elements = all_elements - printed if forward else printed - {element}
            remaining_distance = compute_component_mst(node_points, ground_nodes, remaining_elements,
                                                         initial_position=node_points[second_node])
            #return random.random()
            return tool_distance + remaining_distance
        elif heuristic == 'z':
            return sign * compute_z_distance(node_points, element)
        elif heuristic == 'dijkstra': # offline
            # TODO: sum of all element path distances
            return sign*np.average([distance_from_node[node] for node in element]) # min, max, average
        elif heuristic == 'online-dijkstra':
            if printed not in distance_cache:
                distance_cache[printed] = compute_distance_from_node(printed, node_points, ground_nodes)
            return sign*min(distance_cache[printed].get(node, INF) for node in element)
        elif heuristic == 'plan-stiffness':
            if stiffness_order is None:
                return None
            return sign*stiffness_order[element]
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
