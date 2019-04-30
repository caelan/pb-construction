import heapq
import random

from examples.pybullet.utils.pybullet_tools.utils import connect, ClientSaver, wait_for_user
from extrusion.utils import get_node_neighbors, load_world, check_connected
from extrusion.parsing import draw_element
from extrusion.stream import get_print_gen_fn
from extrusion.sorted import get_z

from collections import namedtuple

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp

State = namedtuple('State', ['element', 'printed', 'plan'])
#Node = namedtuple('State', ['element', 'printed', 'plan'])

def greedy_algorithm(robot, obstacles, node_points, element_bodies, ground_nodes, disable=False):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices

    # stiffness_checker.solve(existing_e_ids)
    # max_t, max_r = stiffness_checker.get_max_nodal_deformation()
    # print("max_t: {0} / {1}, max_r: {2} / {3}".format(max_t, t_tol, max_r, r_tol))
    # t_tol, r_tol = stiffness_checker.get_nodal_deformation_tol()

    # sc = stiffness_checker(json_file_path=file_path, verbose=False)
    # sc.set_self_weight_load(True)
    # sc.set_nodal_displacement_tol(transl_tol=0.003, rot_tol=5 * np.pi / 180)

    # TODO: dynamic programming
    # TODO: max branching factor
    queue = []
    visited = {}
    def add_successor(printed, element, plan):
        if visited in printed:
            return
        priority = (len(printed), get_z(node_points, element), random.random())
        heapq.heappush(queue, (priority, printed, element))

    add_successor(frozenset(element_bodies), None, [])
    while queue:
        _, printed, element = heapq.heappop(queue)
        next_printed = printed if element is None else printed - {element}
        #if not check_connected(ground_nodes, state.printed):
        #    continue
        if not next_printed:
            return
        visited[printed] = plan



        next_plan = []
        for next_element in next_printed: # Sort
            add_successor(next_printed, next_element, next_plan)
    return None
