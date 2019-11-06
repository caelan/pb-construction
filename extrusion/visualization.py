import colorsys

import numpy as np

from extrusion.equilibrium import compute_node_reactions
from extrusion.parsing import load_extrusion
from extrusion.utils import get_node_neighbors, force_from_reaction, is_ground
from pybullet_tools.utils import add_text, draw_pose, get_pose, wait_for_user, add_line, remove_debug, has_gui, \
    draw_point, LockRenderer, set_camera_pose, set_color, apply_alpha, RED, BLUE, GREEN, get_visual_data


def label_element(element_bodies, element):
    element_body = element_bodies[element]
    return [
        add_text(element[0], position=(0, 0, -0.02), parent=element_body),
        add_text(element[1], position=(0, 0, +0.02), parent=element_body),
    ]


def label_elements(element_bodies):
    # +z points parallel to each element body
    for element, body in element_bodies.items():
        print(element)
        label_element(element_bodies, element)
        draw_pose(get_pose(body), length=0.02)
        wait_for_user()

def label_nodes(node_points, **kwargs):
    return [add_text(node, position=point, **kwargs) for node, point in enumerate(node_points)]

def color_structure(element_bodies, printed, next_element):
    # TODO: could also do this with debug segments
    element_colors = {}
    for element in printed:
        element_colors[element] = apply_alpha(BLUE, alpha=1)
    element_colors[next_element] = apply_alpha(GREEN, alpha=1)
    remaining = set(element_bodies) - printed - {next_element}
    for element in remaining:
        element_colors[element] = apply_alpha(RED, alpha=0.5)
    for element, color in element_colors.items():
        body = element_bodies[element]
        [shape] = get_visual_data(body)
        if color != shape.rgbaColor:
            set_color(body, color=color)

def draw_reaction(point, reaction, max_length=0.05, max_force=1, **kwargs):
    vector = max_length * np.array(reaction[:3]) / max_force
    end = point + vector
    return add_line(point, end, **kwargs)


def draw_reactions(node_points, reaction_from_node):
    # TODO: redundant
    handles = []
    for node in sorted(reaction_from_node):
        reactions = reaction_from_node[node]
        max_force = max(map(np.linalg.norm, reactions))
        print('node={}, max force={:.3f}'.format(node, max_force))
        print(list(map(np.array, reactions)))
        start = node_points[node]
        for reaction in reactions:
           handles.append(draw_reaction(start, reaction, max_force=max_force, color=(0, 1, 0)))
        wait_for_user()
        for handle in handles:
            remove_debug(handle)

##################################################

def visualize_stiffness(extrusion_path):
    if not has_gui():
        return
    #label_elements(element_bodies)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    elements = list(element_from_id.values())
    #draw_model(elements, node_points, ground_nodes)

    # Freeform Assembly Planning
    # TODO: https://arxiv.org/pdf/1801.00527.pdf
    # Though assembly sequencing is often done by finding a disassembly sequence and reversing it, we will use a forward search.
    # Thus a low-cost state will usually be correctly identified by considering only the deflection of the cantilevered beam path
    # and approximating the rest of the beams as being infinitely stiff

    reaction_from_node = compute_node_reactions(extrusion_path, elements)
    #reaction_from_node = deformation.displacements # For visualizing displacements
    #test_node_forces(node_points, reaction_from_node)
    force_from_node = {node: sum(np.linalg.norm(force_from_reaction(reaction)) for reaction in reactions)
                       for node, reactions in reaction_from_node.items()}
    sorted_nodes = sorted(reaction_from_node, key=lambda n: force_from_node[n], reverse=True)
    for i, node in enumerate(sorted_nodes):
        print('{}) node={}, point={}, magnitude={:.3E}'.format(
            i, node, node_points[node], force_from_node[node]))

    #max_force = max(force_from_node.values())
    max_force = max(np.linalg.norm(reaction[:3]) for reactions in reaction_from_node.values() for reaction in reactions)
    print('Max force:',  max_force)
    neighbors_from_node = get_node_neighbors(elements)
    colors = sample_colors(len(sorted_nodes))
    handles = []
    for node, color in zip(sorted_nodes, colors):
        #color = (0, 0, 0)
        reactions = reaction_from_node[node]
        #print(np.array(reactions))
        start = node_points[node]
        handles.extend(draw_point(start, color=color))
        for reaction in reactions[:1]:
            handles.append(draw_reaction(start, reaction, max_force=max_force, color=(1, 0, 0)))
        for reaction in reactions[1:]:
            handles.append(draw_reaction(start, reaction, max_force=max_force, color=(0, 1, 0)))
        print('Node: {} | Ground: {} | Neighbors: {} | Reactions: {} | Magnitude: {:.3E}'.format(
            node, (node in ground_nodes), len(neighbors_from_node[node]), len(reactions), force_from_node[node]))
        print('Total:', np.sum(reactions, axis=0))
        wait_for_user()
        #for handle in handles:
        #    remove_debug(handle)
        #handles = []
        #remove_all_debug()

    # TODO: could compute the least balanced node with respect to original forces
    # TODO: sum the norms of all the forces in the structure

    #draw_sequence(sequence, node_points)
    wait_for_user()

##################################################

def draw_element(node_points, element, color=(1, 0, 0)):
    n1, n2 = element
    p1 = node_points[n1]
    p2 = node_points[n2]
    return add_line(p1, p2, color=color[:3])


def draw_model(elements, node_points, ground_nodes):
    handles = []
    with LockRenderer():
        for element in elements:
            color = (0, 0, 1) if is_ground(element, ground_nodes) else (1, 0, 0)
            handles.append(draw_element(node_points, element, color=color))
    return handles


def sample_colors(num, lower=0.0, upper=0.75): # for now wrap around
    return [colorsys.hsv_to_rgb(h, s=1, v=1) for h in np.linspace(lower, upper, num, endpoint=True)]


def draw_ordered(elements, node_points):
    #colors = spaced_colors(len(elements))
    colors = sample_colors(len(elements))
    handles = []
    for element, color in zip(elements, colors):
        handles.append(draw_element(node_points, element, color=color))
    return handles


def set_extrusion_camera(node_points):
    centroid = np.average(node_points, axis=0)
    camera_offset = 0.25 * np.array([1, -1, 1])
    set_camera_pose(camera_point=centroid + camera_offset, target_point=centroid)