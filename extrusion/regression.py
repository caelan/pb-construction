import heapq
import random
import time

from extrusion.progression import Node, sample_extrusion, retrace_trajectories, recover_sequence, \
    recover_directed_sequence, MAX_DIRECTIONS, MAX_ATTEMPTS
from extrusion.heuristics import get_heuristic_fn
from extrusion.motion import compute_motion
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.utils import get_id_from_element, get_ground_elements, create_stiffness_checker, is_ground, \
    check_connected, test_stiffness
from extrusion.validator import compute_plan_deformation
from extrusion.visualization import draw_ordered
from pddlstream.utils import outgoing_from_edges
from pybullet_tools.utils import INF, get_movable_joints, get_joint_positions, randomize, implies, has_gui, \
    remove_all_debug, wait_for_user, elapsed_time


def regression(robot, obstacles, element_bodies, extrusion_path, partial_orders=[],
               heuristic='z', max_time=INF, backtrack_limit=INF,
               collisions=True, stiffness=True, motions=True, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: persistent search
    # TODO: max branching factor

    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    ground_elements = get_ground_elements(all_elements, ground_nodes)
    checker = create_stiffness_checker(extrusion_path, verbose=False)
    #checker = None
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, bidirectional=False, precompute_collisions=False,
                                    max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS,
                                    collisions=collisions, **kwargs)
    heuristic_fn = get_heuristic_fn(extrusion_path, heuristic, checker=checker, forward=False)

    final_conf = initial_conf # TODO: allow choice of config
    final_printed = all_elements
    queue = []
    visited = {final_printed: Node(None, None)}

    outgoing_from_element = outgoing_from_edges(partial_orders)
    def add_successors(printed, conf):
        ground_remaining = printed <= ground_elements
        num_remaining = len(printed) - 1
        assert 0 <= num_remaining
        for element in randomize(printed):
            if outgoing_from_element[element] & printed:
                continue
            if implies(is_ground(element, ground_nodes), ground_remaining):
                bias = heuristic_fn(printed, element, conf=None)
                priority = (num_remaining, bias, random.random())
                heapq.heappush(queue, (priority, printed, element, conf))

    if check_connected(ground_nodes, final_printed) and \
            test_stiffness(extrusion_path, element_from_id, final_printed, checker=checker):
        add_successors(final_printed, final_conf)

    # TODO: lazy hill climbing using the true stiffness heuristic
    # Choose the first move that improves the score
    if has_gui():
        sequence = sorted(final_printed, key=lambda e: heuristic_fn(final_printed, e, conf=None), reverse=True)
        remove_all_debug()
        draw_ordered(sequence, node_points)
        wait_for_user()
    # TODO: fixed branching factor
    # TODO: be more careful when near the end
    # TODO: max time spent evaluating successors (less expensive when few left)
    # TODO: tree rollouts
    # TODO: best-first search with a minimizing path distance cost
    # TODO: immediately select if becomes more stable
    # TODO: focus branching factor on most stable regions

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = transit_failures = 0
    while queue and (elapsed_time(start_time) < max_time):
        priority, printed, element, current_conf = heapq.heappop(queue)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        max_backtrack = max(max_backtrack, backtrack)
        if backtrack_limit < backtrack:
            break # continue
        num_evaluated += 1

        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed - {element}
        #draw_action(node_points, next_printed, element)
        #if 3 < backtrack + 1:
        #    remove_all_debug()
        #    set_renderer(enable=True)
        #    draw_model(next_printed, node_points, ground_nodes)
        #    wait_for_user()

        if (next_printed in visited) or not check_connected(ground_nodes, next_printed) or \
                not implies(stiffness, test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker)):
            continue
        # TODO: could do this eagerly to inspect the full branching factor
        command = sample_extrusion(print_gen_fn, ground_nodes, next_printed, element)
        if command is None:
            continue
        if motions:
            motion_traj = compute_motion(robot, obstacles, element_bodies, node_points, printed,
                                         command.end_conf, current_conf, collisions=collisions,
                                         max_time=max_time - elapsed_time(start_time))
            if motion_traj is None:
                transit_failures += 1
                continue
            command.trajectories.append(motion_traj)

        if num_remaining < min_remaining:
            min_remaining = num_remaining
            print('New best: {}'.format(num_remaining))
            #if has_gui():
            #    # TODO: change link transparency
            #    remove_all_debug()
            #    draw_model(next_printed, node_points, ground_nodes)
            #    wait_for_duration(0.5)

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            plan = retrace_trajectories(visited, next_printed, reverse=True)
            break
        add_successors(next_printed, command.start_conf)

    if plan is not None and motions:
        motion_traj = compute_motion(robot, obstacles, element_bodies, node_points, frozenset(),
                                     initial_conf, plan[0].start_conf, collisions=collisions,
                                     max_time=max_time - elapsed_time(start_time))
        if motion_traj is None:
            transit_failures += 1
            plan = None
        else:
            plan.insert(0, motion_traj)

    max_translation, max_rotation = compute_plan_deformation(extrusion_path, recover_sequence(plan))
    data = {
        'sequence': recover_directed_sequence(plan),
        'runtime': elapsed_time(start_time),
        'num_elements': len(all_elements),
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        'max_translation': max_translation,
        'max_rotation': max_rotation,
        'transit_failures': transit_failures,
    }
    return plan, data