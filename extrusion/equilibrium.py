from collections import namedtuple

import numpy as np

from extrusion.parsing import load_extrusion
from extrusion.utils import get_extructed_ids, create_stiffness_checker, evaluate_stiffness, Reaction, nodes_from_elements
from examples.pybullet.utils.pybullet_tools.utils import get_unit_vector

ReactionForces = namedtuple('Reactions', ['loads', 'fixities', 'reactions'])

def compute_global_reactions(element_from_id, checker, deformation):
    reactions = {}
    local_from_globals = checker.get_element_local2global_rot_matrices()
    for element_id, (start_reaction, end_reaction) in deformation.reactions.items():
        # TODO: simply these computations
        element = element_from_id[element_id]
        global_from_local = np.linalg.inv(local_from_globals[element_id])
        start_rot = global_from_local[:6,:6]
        start_world = np.dot(start_rot, start_reaction)
        end_rot = global_from_local[6:,6:]
        end_world = np.dot(end_rot, end_reaction)
        reactions[element] = (Reaction(*start_world), Reaction(*end_world))
    return reactions

def compute_all_reactions(extrusion_path, elements, checker=None):
    element_from_id, _, _ = load_extrusion(extrusion_path)
    extruded_ids = get_extructed_ids(element_from_id, elements)
    if checker is None:
        checker = create_stiffness_checker(extrusion_path, verbose=False)
    deformation = evaluate_stiffness(extrusion_path, element_from_id, elements, checker=checker, verbose=False)
    # TODO: slight torque due to the load
    nodal_loads = checker.get_nodal_loads(existing_ids=extruded_ids, dof_flattened=False)
    #nodal_loads = checker.get_self_weight_loads(existing_ids=extruded_ids, dof_flattened=False)
    reactions = compute_global_reactions(element_from_id, checker, deformation)
    return ReactionForces(nodal_loads, deformation.fixities, reactions)

def add_reactions(reactions, reaction_from_node):
    for element, (start_reaction, end_reaction) in reactions.items():
        start, end = reversed(element) # TODO: why is this reversed?
        reaction_from_node.setdefault(start, []).append(start_reaction)
        reaction_from_node.setdefault(end, []).append(end_reaction)
    return reaction_from_node

def compute_node_reactions(extrusion_path, elements, **kwargs):
    # https://github.com/yijiangh/conmech/blob/master/tests/test_stiffness_checker.py#L407
    # https://github.com/yijiangh/conmech/blob/master/src/pyconmech/frame_analysis/stiffness_checker.py
    element_from_id, _, ground_nodes = load_extrusion(extrusion_path)
    reaction_forces = compute_all_reactions(extrusion_path, elements, **kwargs)
    loads, fixities, reactions = reaction_forces
    #partial_forces(reaction_forces, elements)

    reaction_from_node = {}
    for node, wrench in loads.items():
        reaction_from_node.setdefault(node, []).append(wrench)
    # The fixities are global. The reaction forces are local
    for node, reaction in fixities.items(): # Fixities are like the ground force to resist the structure?
        reaction_from_node.setdefault(node, []).append(reaction)
    reaction_from_node = add_reactions(reactions, reaction_from_node)
    #nodes = nodes_from_elements(elements) # | ground_nodes
    #assert set(reaction_from_node) == nodes
    return reaction_from_node

def partial_forces(reaction_forces, elements):
    loads, fixities, reactions = reaction_forces
    reaction_from_node = {}
    for node, reaction in fixities.items():
        if node in fixities:
            reaction_from_node[node] = [reaction]
    reactions = {element: reactions[element] for element in elements}
    reaction_from_node = add_reactions(reactions, reaction_from_node)

    for node, reactions in reaction_from_node.items():
        load_force = loads[node][:3]
        total_force = np.sum(reactions, axis=0)[:3]
        unit_load = get_unit_vector(load_force)
        load_magnitude = np.dot(unit_load, load_force)
        total_magnitude = np.dot(unit_load, total_force)
        print(node, load_magnitude, total_magnitude)
    # Partially ordered forward search
