import numpy as np

from extrusion.parsing import load_extrusion
from extrusion.utils import get_extructed_ids, create_stiffness_checker, evaluate_stiffness


def nodal_loads(checker, extruded_ids, reaction_from_node):
    nodal_loads = checker.get_nodal_loads(existing_ids=extruded_ids, dof_flattened=False)
    for node, wrench in nodal_loads.items():
        reaction_from_node.setdefault(node, []).append(wrench)
    #weight_loads = checker.get_self_weight_loads(existing_ids=extruded_ids, dof_flattened=False)
    #for node, wrench in weight_loads.items():
    #    reaction_from_node.setdefault(node, []).append(wrench)


def ground_reactions(deformation, reaction_from_node):
    # The fixities are global. The reaction forces are local
    for node, reaction in deformation.fixities.items(): # Fixities are like the ground force to resist the structure?
        reaction_from_node.setdefault(node, []).append(reaction)


def local_reactions(extrusion_path, elements):
    # https://github.com/yijiangh/conmech/blob/master/tests/test_stiffness_checker.py#L407
    # https://github.com/yijiangh/conmech/blob/master/src/pyconmech/frame_analysis/stiffness_checker.py
    element_from_id, _, _ = load_extrusion(extrusion_path)
    extruded_ids = get_extructed_ids(element_from_id, elements)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    deformation = evaluate_stiffness(extrusion_path, element_from_id, elements, verbose=False)

    reaction_from_node = {}
    nodal_loads(checker, extruded_ids, reaction_from_node)
    ground_reactions(deformation, reaction_from_node)

    local_from_globals = checker.get_element_local2global_rot_matrices()
    for element_id, (start_reaction, end_reaction) in deformation.reactions.items():
        element = element_from_id[element_id]
        start, end = reversed(element)

        global_from_local = np.linalg.inv(local_from_globals[element_id])
        start_rot = global_from_local[:6,:6]
        #start_force = quat_from_matrix(start_rot[:3,:3])
        #start_torque = quat_from_matrix(start_rot[3:6,3:6])
        #assert np.allclose(start_force, start_torque)

        end_rot = global_from_local[6:,6:]
        #end_force = quat_from_matrix(end_rot[:3,:3])
        #end_torque = quat_from_matrix(end_rot[3:,3:])
        #assert np.allclose(start_force, end_force)
        #assert np.allclose(start_torque, end_torque)

        start_world = np.dot(start_rot, start_reaction)
        reaction_from_node.setdefault(start, []).append(start_world)
        end_world = np.dot(end_rot, end_reaction)
        reaction_from_node.setdefault(end, []).append(end_world)
    return reaction_from_node