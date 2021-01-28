import heapq
import random
import time

from extrusion.progression import Node, retrace_commands
from extrusion.heuristics import get_heuristic_fn, get_tool_position
from extrusion.motion import compute_motion, compute_motions, LAZY
from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn
from extrusion.utils import get_id_from_element, get_ground_elements, is_ground, \
    check_connected, check_memory, get_undirected, get_directions, compute_printed_nodes, \
    flatten_commands
from extrusion.stiffness import create_stiffness_checker, test_stiffness
from extrusion.visualization import draw_element
from pddlstream.utils import outgoing_from_edges
from pybullet_tools.utils import INF, randomize, has_gui, \
    remove_all_debug, wait_for_user, elapsed_time, implies, LockRenderer, get_configuration
from extrusion.optimize import optimize_commands, OPTIMIZE


def draw_action(node_points, printed, element):
    if not has_gui():
        return []
    with LockRenderer():
        remove_all_debug()
        handles = [draw_element(node_points, element, color=(1, 0, 0))]
        handles.extend(draw_element(node_points, e, color=(0, 1, 0)) for e in printed)
    wait_for_user()
    return handles

##################################################

def regression(robot, obstacles, element_bodies, extrusion_path, partial_orders=[],
               heuristic='z', max_time=INF, max_memory=INF, backtrack_limit=INF, revisit=False, stiffness_attempts=1,
               collisions=True, stiffness=True, motions=True, lazy=LAZY, checker=None, **kwargs):
    # Focused has the benefit of reusing prior work
    # Greedy has the benefit of conditioning on previous choices
    # TODO: max branching factor
    # TODO: be more careful when near the end
    # TODO: max time spent evaluating successors (less expensive when few left)
    # TODO: tree rollouts
    # TODO: best-first search with a minimizing path distance cost
    # TODO: immediately select if becomes more stable
    # TODO: focus branching factor on most stable regions

    start_time = time.time()
    initial_conf = get_configuration(robot)
    initial_position = get_tool_position(robot)
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    id_from_element = get_id_from_element(element_from_id)
    all_elements = frozenset(element_bodies)
    ground_elements = get_ground_elements(all_elements, ground_nodes)
    if stiffness and (checker is None):
        checker = create_stiffness_checker(extrusion_path, verbose=False)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    precompute_collisions=False, collisions=collisions, **kwargs)
    heuristic_fn = get_heuristic_fn(robot, extrusion_path, heuristic, checker=checker, forward=False)

    queue = []
    outgoing_from_element = outgoing_from_edges(partial_orders)
    def add_successors(printed, position, conf):
        only_ground = printed <= ground_elements
        num_remaining = len(printed) - 1
        #assert 0 <= num_remaining
        for element in randomize(printed):
            if not (outgoing_from_element[element] & printed) and implies(is_ground(element, ground_nodes), only_ground):
                for directed in get_directions(element):
                    visits = 0
                    bias = heuristic_fn(printed, directed, position, conf)
                    priority = (num_remaining, bias, random.random())
                    heapq.heappush(queue, (visits, priority, printed, directed, conf))

    final_conf = initial_conf # TODO: allow choice of final config
    final_position = initial_position
    final_printed = all_elements
    visited = {final_printed: Node(None, None)}
    if check_connected(ground_nodes, final_printed) and \
            (not stiffness or test_stiffness(extrusion_path, element_from_id, final_printed, checker=checker)):
        add_successors(final_printed, final_position, final_conf)

    # if has_gui():
    #     sequence = sorted(final_printed, key=lambda e: heuristic_fn(final_printed, e, conf=None), reverse=True)
    #     remove_all_debug()
    #     draw_ordered(sequence, node_points)
    #     wait_for_user()

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = extrusion_failures = transit_failures = stiffness_failures = 0
    while queue and (elapsed_time(start_time) < max_time) and check_memory(): #max_memory):
        visits, priority, printed, directed, current_conf = heapq.heappop(queue)
        element = get_undirected(all_elements, directed)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        max_backtrack = max(max_backtrack, backtrack)
        if backtrack_limit < backtrack:
            break # continue
        num_evaluated += 1

        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Index: {} | Time: {:.3f}'.format(
            num_evaluated, min_remaining, len(printed), element, id_from_element[element], elapsed_time(start_time)))
        next_printed = printed - {element}
        next_nodes = compute_printed_nodes(ground_nodes, next_printed)

        #draw_action(node_points, next_printed, element)
        #if 3 < backtrack + 1:
        #    remove_all_debug()
        #    set_renderer(enable=True)
        #    draw_model(next_printed, node_points, ground_nodes)
        #    wait_for_user()

        node1, node2 = directed
        if (next_printed in visited) or (node1 not in next_nodes) or not check_connected(ground_nodes, next_printed):
            continue
        # TODO: stiffness plan lazily here possibly with reuse
        if stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker):
            stiffness_failures += 1
            continue
        # TODO: stronger condition for this procedure
        # if plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, next_printed,
        #                   checker=checker, max_backtrack=0) is None:
        #     # TODO: cache and reuse prior stiffness plans
        #     print('Failed stiffness plan') # TODO: require just a short horizon
        #     continue
        if revisit:
            heapq.heappush(queue, (visits + 1, priority, printed, directed, current_conf))

        command, = next(print_gen_fn(node1, element, extruded=next_printed), (None,))
        if command is None:
            extrusion_failures += 1
            continue
        if motions and not lazy:
            motion_traj = compute_motion(robot, obstacles, element_bodies, printed,
                                         command.end_conf, current_conf, collisions=collisions,
                                         max_time=max_time - elapsed_time(start_time)) # TODO: smooth=...)
            if motion_traj is None:
                transit_failures += 1
                continue
            command.trajectories.append(motion_traj)

        if num_remaining < min_remaining:
            min_remaining = num_remaining
            #print('New best: {}'.format(num_remaining))
            #if has_gui():
            #    # TODO: change link transparency
            #    remove_all_debug()
            #    draw_model(next_printed, node_points, ground_nodes)
            #    wait_for_duration(0.5)

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            commands = retrace_commands(visited, next_printed, reverse=True)
            if OPTIMIZE:
                commands = optimize_commands(robot, obstacles, element_bodies, extrusion_path, initial_conf, commands,
                                             motions=motions, collisions=collisions)
            plan = flatten_commands(commands)
            if motions and not lazy:
                motion_traj = compute_motion(robot, obstacles, element_bodies, frozenset(),
                                             initial_conf, plan[0].start_conf, collisions=collisions,
                                             max_time=max_time - elapsed_time(start_time))
                if motion_traj is None:
                    plan = None
                    transit_failures += 1
                else:
                    plan.insert(0, motion_traj)
            if motions and lazy:
                plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan,
                                       collisions=collisions, max_time=max_time - elapsed_time(start_time))
            break
            # if plan is not None:
            #     break
        add_successors(next_printed, node_points[node1], command.start_conf)
    #del checker

    data = {
        #'memory': get_memory_in_kb(), # May need to update instead
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        'stiffness_failures': stiffness_failures,
        'extrusion_failures': extrusion_failures,
        'transit_failures': transit_failures,
    }
    return plan, data
