import numpy as np

from extrusion.equilibrium import compute_node_reactions
from extrusion.parsing import load_extrusion, sample_colors
from extrusion.utils import get_node_neighbors
from examples.pybullet.utils.pybullet_tools.utils import add_text, draw_pose, get_pose, wait_for_user, add_line, remove_debug, has_gui, \
    draw_point


def label_nodes(element_bodies, element):
    element_body = element_bodies[element]
    return [
        add_text(element[0], position=(0, 0, -0.02), parent=element_body),
        add_text(element[1], position=(0, 0, +0.02), parent=element_body),
    ]


def label_elements(element_bodies):
    # +z points parallel to each element body
    for element, body in element_bodies.items():
        print(element)
        label_nodes(element_bodies, element)
        draw_pose(get_pose(body), length=0.02)
        wait_for_user()


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
    total_reaction_from_node = {node: np.sum(reactions, axis=0)[:3]
                                for node, reactions in reaction_from_node.items()}
    force_from_node = {node: np.linalg.norm(reaction)
                       for node, reaction in total_reaction_from_node.items()}
    #max_force = max(force_from_node.values())
    max_force = max(np.linalg.norm(reaction[:3]) for reactions in reaction_from_node.values() for reaction in reactions)
    print('Max force:',  max_force)
    for i, node in enumerate(sorted(total_reaction_from_node, key=lambda n: force_from_node[n])):
        print('{}) node={}, point={}, vector={}, magnitude={:.3E}'.format(
            i, node, node_points[node], total_reaction_from_node[node], force_from_node[node]))

    neighbors_from_node = get_node_neighbors(elements)
    nodes = sorted(reaction_from_node, key=lambda n: force_from_node[n])
    colors = sample_colors(len(nodes))
    handles = []
    for node, color in zip(nodes, colors):
        color = (0, 0, 0)
        reactions = reaction_from_node[node]
        #print(np.array(reactions))
        start = node_points[node]
        handles.extend(draw_point(start, color=color))
        for reaction in reactions[:1]:
            handles.append(draw_reaction(start, reaction, max_force=max_force, color=(1, 0, 0)))
        for reaction in reactions[1:]:
            handles.append(draw_reaction(start, reaction, max_force=max_force, color=(0, 1, 0)))
        print('Node: {} | Ground: {} | Neighbors: {} | Reactions: {}'.format(
            node, (node in ground_nodes), len(neighbors_from_node[node]), len(reactions)))
        print(np.sum(reactions, axis=0))
        #handles.append(draw(start, total_reaction_from_node[node], max_force=max_force, color=(0, 0, 1)))
        wait_for_user()
        #for handle in handles:
        #    remove_debug(handle)
        #handles = []
        #remove_all_debug()

    # TODO: could compute the least balanced node with respect to original forces
    # TODO: sum the norms of all the forces in the structure

    #draw_sequence(sequence, node_points)
    wait_for_user()