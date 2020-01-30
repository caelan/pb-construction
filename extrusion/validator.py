import numpy as np
from extrusion.parsing import load_extrusion, extrusion_name_from_path, get_extrusion_path
from extrusion.visualization import draw_element
from extrusion.utils import check_connected, get_connected_structures, load_world
from extrusion.stiffness import TRANS_TOL, ROT_TOL, create_stiffness_checker, evaluate_stiffness, test_stiffness
from pybullet_tools.utils import has_gui, wait_for_user, connect, reset_simulation, \
    disconnect, wait_for_duration, BLACK, RED


def check_plan(extrusion_path, planned_elements, verbose=False):
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    #checker = create_stiffness_checker(extrusion_name)

    connected_nodes = set(ground_nodes)
    handles = []
    all_connected = True
    all_stiff = True
    extruded_elements = set()
    for element in planned_elements:
        extruded_elements.add(element)
        n1, n2 = element
        #is_connected &= any(n in connected_nodes for n in element)
        is_connected = (n1 in connected_nodes)
        connected_nodes.update(element)
        #is_connected = check_connected(ground_nodes, extruded_elements)
        #all_connected &= is_connected
        is_stiff = test_stiffness(extrusion_path, element_from_id, extruded_elements, verbose=verbose)
        all_stiff &= is_stiff
        if verbose:
            structures = get_connected_structures(extruded_elements)
            print('Elements: {} | Structures: {} | Connected: {} | Stiff: {}'.format(
                len(extruded_elements), len(structures), is_connected, is_stiff))
        if has_gui():
            is_stable = is_connected and is_stiff
            color = BLACK if is_stable else RED
            handles.append(draw_element(node_points, element, color))
            wait_for_duration(0.1)
            if not is_stable:
                wait_for_user()
    # Make these counts instead
    print('Connected: {} | Stiff: {}'.format(all_connected, all_stiff))
    return all_connected and all_stiff


def verify_plan(extrusion_path, planned_elements, use_gui=False, **kwargs):
    # Path heuristic
    # Disable shadows
    if use_gui:
        connect(use_gui=use_gui)
        obstacles, robot = load_world()
    is_valid = check_plan(extrusion_path, planned_elements, **kwargs)
    if use_gui:
        reset_simulation()
        disconnect()
    return is_valid

##################################################

def compute_plan_deformation(problem, plan):
    # TODO: absence of entry means ignore
    problem = extrusion_name_from_path(problem)
    problem_path = get_extrusion_path(problem)
    checker = create_stiffness_checker(problem_path, verbose=False)
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    if plan is None:
        return (-1, -1), (-1, -1), (-1, -1)

    element_from_id, _, _ = load_extrusion(problem_path)
    printed = []
    translations = []
    rotations = []
    compliances = []
    for element in plan:
        printed.append(element)
        deformation = evaluate_stiffness(problem, element_from_id, printed,
                                         checker=checker, verbose=False)
        displacements = deformation.displacements 
        trans = np.max(np.linalg.norm([d[:3] for d in displacements.values()], ord=2, axis=1))
        rot = np.max(np.linalg.norm([d[3:] for d in displacements.values()], ord=2, axis=1))
        # trans, rot, _, _ = checker.get_max_nodal_deformation()
        translations.append(trans)
        rotations.append(rot)
        compliances.append(deformation.compliance)
    # TODO: could return full history
    return (max(translations), int(np.argmax(translations))), (max(rotations), int(np.argmax(rotations))), \
           (max(compliances), int(np.argmax(compliances))) # larger compliance means more flexible
