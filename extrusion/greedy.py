import heapq
import random
import numpy as np

from examples.pybullet.utils.pybullet_tools.utils import connect, ClientSaver, wait_for_user
from extrusion.utils import get_node_neighbors, load_world, check_connected, check_stiffness
from extrusion.parsing import draw_element, load_extrusion, get_extrusion_path
from extrusion.stream import get_print_gen_fn
from extrusion.sorted import get_z
from pyconmech import stiffness_checker

from collections import namedtuple

# https://github.com/yijiangh/conmech/blob/master/src/bindings/pyconmech/pyconmech.cpp

State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])

def retrace_plan(visited, current_state):
    command, prev_state = visited[current_state]
    if prev_state is None:
        return []
    return [traj for traj in command.trajectories] + retrace_plan(visited, prev_state)

def greedy_algorithm(robot, obstacles, element_bodies, extrusion_name, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices

    extrusion_path = get_extrusion_path(extrusion_name)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_name)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, max_attempts=1000, **kwargs)
    # TODO: persistent search to reuse
    # stiffness_checker.solve(existing_e_ids)
    # max_t, max_r = stiffness_checker.get_max_nodal_deformation()
    # t_tol, r_tol = stiffness_checker.get_nodal_deformation_tol()
    # print("max_t: {0} / {1}, max_r: {2} / {3}".format(max_t, t_tol, max_r, r_tol))

    sc = stiffness_checker(json_file_path=extrusion_path, verbose=False)
    sc.set_self_weight_load(True)
    sc.set_nodal_displacement_tol(transl_tol=0.003, rot_tol=5 * np.pi / 180)

    # sc.set_output_json(True)
    # sc.set_output_json_path(file_path = cwd, file_name = "sf-test_result.json")

    # orig_beam_shape = sc.get_original_shape(disc=disc, draw_full_shape=False)
    # beam_disp = sc.get_deformed_shape(exagg_ratio=exagg_ratio, disc=disc)

    # TODO: dynamic programming
    # TODO: max branching factor
    queue = []
    visited = {}
    def add_successors(printed):
        for element in sorted(printed, key=lambda e: -get_z(node_points, e)):
            priority = (len(printed), -get_z(node_points, element), random.random())
            heapq.heappush(queue, (priority, printed, element))

    initial_printed = frozenset(element_bodies)
    if not check_connected(ground_nodes, initial_printed) or \
            not check_stiffness(sc, element_from_id, initial_printed):
        return None
    visited[initial_printed] = Node(None, None)
    add_successors(initial_printed)

    iteration = 0
    while queue:
        # TODO: persistent search
        iteration += 1
        _, printed, element = heapq.heappop(queue)
        print('Iteration: {} | Printed: {} | Element: {}'.format(iteration, len(printed), element))
        next_printed = printed - {element}
        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not check_stiffness(sc, element_from_id, next_printed):
            continue

        next_nodes = {n for e in next_printed for n in e} | set(ground_nodes)
        for node in element:
            if node in next_nodes:
                try:
                    command, = next(print_gen_fn(node, element, extruded=next_printed))
                    break
                except StopIteration:
                    pass
        else:
            continue
        visited[next_printed] = Node(command, printed)
        if not next_printed:
            return retrace_plan(visited, next_printed)
        add_successors(next_printed)
    return None
