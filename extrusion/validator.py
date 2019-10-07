from extrusion.parsing import load_extrusion, draw_element
from extrusion.utils import check_connected, test_stiffness, get_connected_structures, load_world
from pybullet_tools.utils import has_gui, wait_for_user, connect, reset_simulation, \
    disconnect, wait_for_duration


def check_plan(extrusion_path, planned_elements, verbose=False):
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    #checker = create_stiffness_checker(extrusion_name)

    # TODO: construct the structure in different ways (random, connected)
    handles = []
    all_connected = True
    all_stiff = True
    extruded_elements = set()
    for element in planned_elements:
        extruded_elements.add(element)
        is_connected = check_connected(ground_nodes, extruded_elements)
        all_connected &= is_connected
        is_stiff = test_stiffness(extrusion_path, element_from_id, extruded_elements)
        all_stiff &= is_stiff
        if verbose:
            structures = get_connected_structures(extruded_elements)
            print('Elements: {} | Structures: {} | Connected: {} | Stiff: {}'.format(
                len(extruded_elements), len(structures), is_connected, is_stiff))
        if has_gui():
            is_stable = is_connected and is_stiff
            color = (0, 1, 0) if is_stable else (1, 0, 0)
            handles.append(draw_element(node_points, element, color))
            wait_for_duration(0.5)
            if not is_stable:
                wait_for_user()
    # Make these counts instead
    print('Connected: {} | Stiff: {}'.format(all_connected, all_stiff))
    return all_connected and all_stiff


def verify_plan(extrusion_path, planned_elements, use_gui=False):
    # Path heuristic
    # Disable shadows
    connect(use_gui=use_gui)
    obstacles, robot = load_world()
    is_valid = check_plan(extrusion_path, planned_elements)
    reset_simulation()
    disconnect()
    return is_valid
