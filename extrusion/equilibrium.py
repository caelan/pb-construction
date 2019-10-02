import numpy as np

from extrusion.parsing import load_extrusion
from extrusion.utils import get_extructed_ids, create_stiffness_checker, evaluate_stiffness, Reaction

def compute_global_reactions(checker, deformation):
    reactions = {}
    local_from_globals = checker.get_element_local2global_rot_matrices()
    for element_id, (start_reaction, end_reaction) in deformation.reactions.items():
        # TODO: simply these computations
        global_from_local = np.linalg.inv(local_from_globals[element_id])
        start_rot = global_from_local[:6,:6]
        start_world = np.dot(start_rot, start_reaction)
        end_rot = global_from_local[6:,6:]
        end_world = np.dot(end_rot, end_reaction)
        reactions[element_id] = (start_world, end_world)
    return reactions

def compute_all_reactions(extrusion_path, elements, checker=None):
    element_from_id, _, _ = load_extrusion(extrusion_path)
    extruded_ids = get_extructed_ids(element_from_id, elements)
    if checker is None:
        checker = create_stiffness_checker(extrusion_path, verbose=False)
    deformation = evaluate_stiffness(extrusion_path, element_from_id, elements, checker=checker, verbose=False)
    nodal_loads = checker.get_nodal_loads(existing_ids=extruded_ids, dof_flattened=False)
    #weight_loads = checker.get_self_weight_loads(existing_ids=extruded_ids, dof_flattened=False)
    reactions = compute_global_reactions(checker, deformation)
    return nodal_loads, deformation.fixities, reactions

def compute_node_reactions(extrusion_path, elements, **kwargs):
    # https://github.com/yijiangh/conmech/blob/master/tests/test_stiffness_checker.py#L407
    # https://github.com/yijiangh/conmech/blob/master/src/pyconmech/frame_analysis/stiffness_checker.py
    element_from_id, _, _ = load_extrusion(extrusion_path)
    nodal_loads, fixities, reactions = compute_all_reactions(extrusion_path, elements, **kwargs)

    reaction_from_node = {}
    for node, wrench in nodal_loads.items():
        reaction_from_node.setdefault(node, []).append(wrench)
    # The fixities are global. The reaction forces are local
    for node, reaction in fixities.items(): # Fixities are like the ground force to resist the structure?
        reaction_from_node.setdefault(node, []).append(reaction)
    for element_id, (start_reaction, end_reaction) in reactions.items():
        element = element_from_id[element_id]
        start, end = reversed(element) # TODO: why is this reversed?
        reaction_from_node.setdefault(start, []).append(start_reaction)
        reaction_from_node.setdefault(end, []).append(end_reaction)
    return reaction_from_node
