import heapq
import random

from examples.pybullet.utils.pybullet_tools.utils import connect, ClientSaver, wait_for_user
from extrusion.utils import get_node_neighbors, load_world
from extrusion.parsing import draw_element
from extrusion.stream import get_print_gen_fn


def get_z(node_points, element):
    return max(node_points[n][2] for n in element)

def heuristic_planner(robot, obstacles, node_points, element_bodies, ground_nodes, **kwargs):
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, **kwargs)
    node_neighbors = get_node_neighbors(element_bodies)

    connected_nodes = set()
    queue = []
    queued = set()
    def add_node(node):
        connected_nodes.add(node)
        for element in node_neighbors[node]:
            if element not in queued:
                score = (get_z(node_points, element), random.random())
                heapq.heappush(queue, (score, element))
                queued.add(element)

    for node in ground_nodes:
        add_node(node)
    extruded_elements = set()
    planned_trajectories = []
    while queue:
        _, element = heapq.heappop(queue)
        # TODO: tiebreak by angle or x
        for node in element:
            try:
                trajectory, = next(print_gen_fn(node, element, extruded=extruded_elements))
                planned_trajectories.append(trajectory)
                extruded_elements.add(element)
                break
            except StopIteration:
                pass
        else:
            #return None
            client = connect(use_gui=True)
            with ClientSaver(client):
                floor, robot = load_world()
                handles = []
                for e in extruded_elements:
                    handles.append(draw_element(node_points, e, color=(0, 1, 0)))
                handles.append(draw_element(node_points, element, color=(1, 0, 0)))
                print('Failure!')
                wait_for_user()
            return None
        for node in element:
            add_node(node)
    return planned_trajectories