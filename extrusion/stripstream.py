from __future__ import print_function

import time
import os

from extrusion.decomposition import compute_total_orders, extract_static_facts
from extrusion.fluent import get_location_distance, get_test_printable, get_test_stiff, get_fluent_print_gen_fn, \
    get_order_fn
from extrusion.heuristics import compute_layer_from_vertex
from extrusion.temporal import compute_directions, compute_local_orders, compute_elements_from_layer, \
    compute_global_orders, simulate_parallel, compute_assignments, compute_transits, get_opt_distance_fn, \
    ROBOT_TEMPLATE, mirror_robot, GREEDY_PLANNER, POSTPROCESS_PLANNER, get_wild_move_gen_fn, get_wild_print_gen_fn, \
    get_collision_test, Conf
from extrusion.utils import get_element_length, TOOL_VELOCITY, flatten_commands, nodes_from_elements
from extrusion.visualization import draw_model
#from examples.pybullet.turtlebots.run import *

from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.focused import solve_focused #, CURRENT_STREAM_PLAN
from pddlstream.language.constants import And, PDDLProblem, print_solution, DurativeAction, Equal, print_plan
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read, get_file_path
from pddlstream.language.temporal import compute_duration, compute_start, compute_end, apply_start, \
    DURATIVE_ACTIONS, reverse_plan, get_tfd_path

from pybullet_tools.utils import has_gui, WorldSaver, RED, \
    INF, LockRenderer, SEPARATOR, user_input, remove_all_debug, GREEN, elapsed_time

STRIPSTREAM_ALGORITHM = 'stripstream' # focused, incremental

CUSTOM_LIMITS = { # TODO: do instead of modifying the URDF
   #'robot_joint_a1': (-np.pi/2, np.pi/2),
   #'robot_joint_a1': (-np.pi / 2, 0),
}

# ----- Small -----
# extreme_beam_test
# extrusion_exp_L75.0
# four-frame
# simple_frame
# topopt-205_long_beam_test
# long_beam_test

# ----- Medium -----
# robarch_tree_S, robarch_tree_M
# topopt-101_tiny
# semi_sphere
# compas_fea_beam_tree_simp, compas_fea_beam_tree_S_simp, compas_fea_beam_tree_M_simp


##################################################

def get_pddlstream(robots, static_obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                   initial_confs={}, printed=set(), removed=set(),
                   additional_init=[], additional_orders=set(), trajectories=[],
                   temporal=True, sequential=False, local=False,
                   can_print=True, can_transit=False,
                   checker=None, **kwargs):
    try:
        get_tfd_path()
    except RuntimeError:
        temporal = False
        print('Temporal Fast Downward is not installed. Disabling temporal planning.')
    # TODO: TFD submodule
    assert not removed & printed
    remaining = set(element_bodies) - removed - printed
    element_obstacles = {element_bodies[e] for e in printed}
    obstacles = set(static_obstacles) | element_obstacles
    max_layer = max(layer_from_n.values())

    directions = compute_directions(remaining, layer_from_n)
    if local:
        # makespan seems more effective than CEA
        partial_orders = compute_local_orders(remaining, layer_from_n) # makes the makespan heuristic slow
    else:
        partial_orders = compute_global_orders(remaining, layer_from_n)
    partial_orders.update(additional_orders)
    # draw_model(supporters, node_points, ground_nodes, color=RED)
    # wait_if_gui()

    use_fluent = True
    if use_fluent:
        domain_file = 'domain_fluent.pddl'
        stream_file = 'stream_fluent.pddl'
    else:
        domain_file = 'temporal.pddl' if temporal else 'domain.pddl'
        stream_file = 'stream.pddl'
    domain_pddl = read(get_file_path(__file__, os.path.join('pddl', domain_file)))
    stream_pddl = read(get_file_path(__file__, os.path.join('pddl', stream_file)))
    constant_map = {}

    # TODO: don't evaluate TrajTrajCollision until the plan is retimed
    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-move': get_wild_move_gen_fn(robots, obstacles, element_bodies,
                                            partial_orders=partial_orders, **kwargs),
        'sample-print': get_wild_print_gen_fn(robots, obstacles, node_points, element_bodies, ground_nodes,
                                              initial_confs=initial_confs, partial_orders=partial_orders,
                                              removed=removed, **kwargs),

        'test-printable': from_test(get_test_printable(ground_nodes)),
        'test-stiff': from_test(get_test_stiff()),
        'LocationDistance': get_location_distance(node_points, robots, initial_confs),

        #'test-cfree-traj-conf': from_test(lambda *args: True),
        #'test-cfree-traj-traj': from_test(get_cfree_test(**kwargs)),

        'TrajTrajCollision': get_collision_test(robots, **kwargs),
        'TrajConfCollision': lambda *args, **kwargs: False, # TODO: could treat conf as length 1 traj

        'Length': lambda e: get_element_length(e, node_points),
        'Distance': lambda r, t: t.get_link_distance(),
        'Duration': lambda r, t: t.get_link_distance() / TOOL_VELOCITY,
        'Euclidean': lambda n1, n2: get_element_length((n1, n2), node_points),
    }
    if use_fluent:
        stream_map.update({
            'sample-print': from_gen_fn(get_fluent_print_gen_fn(
                robots, obstacles, node_points, element_bodies, ground_nodes,
                partial_orders=partial_orders, removed=removed, **kwargs)),
        })

    assignments = compute_assignments(robots, remaining, node_points, initial_confs)
    transits = compute_transits(layer_from_n, directions)

    init = [
        ('Stationary',),
        #Equal(('PrintCost',), 1),
        Equal(('Speed',), TOOL_VELOCITY),
    ]
    init.extend(additional_init)
    if can_print:
        init.append(('Print',)) # TODO: Printable and Movable per robot
    if can_transit:
        init.append(('Move',))
    if sequential:
        init.append(('Sequential',))

    for name, conf in initial_confs.items():
        #robot = index_from_name(robots, name)
        #init_loc = -robot
        init_loc = '{}-q0'.format(name)
        init.extend([
            ('Location', init_loc),
            ('AtLoc', name, init_loc),
            ('BackoffConf', name, conf),
            ('Robot', name),
            ('Conf', name, conf),
            ('AtConf', name, conf),
            ('Idle', name),
            #('Start', name, init_loc, None, conf),
            #('End', name, None, init_loc, conf),
        ])
        if can_transit:
            init.append(('CanMove', name)) # TODO: might need to comment out again
        for (n1, e, n2) in directions:
            if layer_from_n[n1] == 0:
                transits.append((None, init_loc, n1, e))
            if layer_from_n[n2] == max_layer:
                transits.append((e, n2, init_loc, None))

    init.extend(('Grounded', n) for n in ground_nodes)
    init.extend(('Direction',) + tup for tup in directions) # Directed edge
    init.extend(('Order',) + tup for tup in partial_orders)
    # TODO: can relax assigned if I go by layers
    init.extend(('Assigned', r, e) for r in assignments for e in assignments[r])
    #init.extend(('Transit',) + tup for tup in transits)
    # TODO: only move actions between adjacent layers

    for n in nodes_from_elements(remaining):
        init.extend([
            ('Node', n),
            ('Location', n),
        ])
    for e in remaining:
        init.extend([
            ('Element', e),
            ('Printed', e),
        ])
        #n1, n2 = ['n{}'.format(i) for i in e]
        for n1, n2 in [e, reversed(e)]:
            init.extend([
                ('Endpoint', n1, e),
                ('Edge', n1, e, n2),
            ])

    assert not trajectories
    # for t in trajectories:
    #     init.extend([
    #         ('Traj', t),
    #         ('PrintAction', t.n1, t.element, t),
    #     ])

    goal_literals = []
    #if can_transit:
    #    goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items()) # TODO: AtLoc
    goal_literals.extend(('Removed', e) for e in remaining)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################

def solve_pddlstream(problem, node_points, element_bodies, planner=GREEDY_PLANNER, max_time=60):
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    # TODO: only consider axioms that could be relevant
    # TODO: iterated search using random restarts
    # TODO: most of the time seems to be spent extracting the stream plan
    # TODO: NEGATIVE_SUFFIX to make axioms easier
    # TODO: sort by action cost heuristic
    # http://www.fast-downward.org/Doc/Evaluator#Max_evaluator

    temporal = DURATIVE_ACTIONS in problem.domain_pddl
    print('Init:', problem.init)
    print('Goal:', problem.goal)
    print('Max time:', max_time)
    print('Temporal:', temporal)

    stream_info = {
        # TODO: stream effort
        'sample-print': StreamInfo(PartialInputs(unique=True)),
        'sample-move': StreamInfo(PartialInputs(unique=True)),

        'test-cfree-traj-conf': StreamInfo(p_success=1e-2, negate=True),  # , verbose=False),
        'test-cfree-traj-traj': StreamInfo(p_success=1e-2, negate=True),
        'TrajConfCollision': FunctionInfo(p_success=1e-1, overhead=1),  # TODO: verbose
        'TrajTrajCollision': FunctionInfo(p_success=1e-1, overhead=1),  # TODO: verbose

        'Distance': FunctionInfo(opt_fn=get_opt_distance_fn(element_bodies, node_points), eager=True)
        # 'Length': FunctionInfo(eager=True),  # Need to eagerly evaluate otherwise 0 makespan (failure)
        # 'Duration': FunctionInfo(opt_fn=lambda r, t: opt_distance / TOOL_VELOCITY, eager=True),
        # 'Euclidean': FunctionInfo(eager=True),
    }

    # TODO: goal serialization
    # TODO: could revert back to goal count now that no deadends
    # TODO: limit the branching factor if necessary
    # TODO: ensure that function costs aren't prunning plans
    if not temporal:
        # Reachability heuristics good for detecting dead-ends
        # Infeasibility from the start means disconnected or collision
        set_cost_scale(1)
        # planner = 'ff-ehc'
        # planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
        planner = 'ff-eager-tiebreak'  # Need to use a eager search, otherwise doesn't incorporate child cost
        # planner = 'max-astar'

    # TODO: assert (instance.value == value)
    use_incremental = use_attachments = True
    with LockRenderer(lock=False):
        if use_incremental:
            if use_attachments:
                planner = {
                    'search': 'lazy', # eager | lazy
                    'evaluator': 'greedy',
                    'heuristic': 'ff', # goal | ff (can detect dead ends)
                    #'heuristic': ['ff', get_bias_fn(element_from_index)],
                    #'successors': 'all',
                    'successors': get_order_fn(node_points), # TODO: confirm that this is working correctly
                }
            else:
                planner = 'add-random-lazy'
            solution = solve_incremental(problem, planner=planner, max_time=600,
                                         max_planner_time=300, debug=True, verbose=True)
        else:
            # TODO: allow some types of failures
            solution = solve_focused(problem, stream_info=stream_info, max_time=max_time,
                                     effort_weight=None, unit_efforts=True, unit_costs=False, # TODO: effort_weight=None vs 0
                                     max_skeletons=None, bind=True, max_failures=INF,  # 0 | INF
                                     planner=planner, max_planner_time=60, debug=False, reorder=False,
                                     initial_complexity=1)

    print_solution(solution)
    plan, _, certificate = solution
    # TODO: post-process by calling planner again
    # TODO: could solve for trajectories conditioned on the sequence
    return plan, certificate

##################################################

def solve_joint(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                trajectories=[], collisions=True, disable=False, max_time=INF, **kwargs):
    problem = get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                             trajectories=trajectories, collisions=collisions, disable=disable, **kwargs)
    return solve_pddlstream(problem, node_points, element_bodies, max_time=max_time)

def solve_serialized(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                     trajectories=[], post_process=False, collisions=True, disable=False, max_time=INF, **kwargs):
    start_time = time.time()
    saver = WorldSaver()
    elements = set(element_bodies)
    elements_from_layers = compute_elements_from_layer(elements, layer_from_n)
    layers = sorted(elements_from_layers.keys())
    print('Layers:', layers)

    full_plan = []
    makespan = 0.
    removed = set()
    for layer in reversed(layers):
        print(SEPARATOR)
        print('Layer: {}'.format(layer))
        saver.restore()
        remaining = elements_from_layers[layer]
        printed = elements - remaining - removed
        draw_model(remaining, node_points, ground_nodes, color=GREEN)
        draw_model(printed, node_points, ground_nodes, color=RED)
        problem = get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                                 printed=printed, removed=removed, return_home=False,
                                 trajectories=trajectories, collisions=collisions, disable=disable, **kwargs)
        layer_plan, certificate = solve_pddlstream(problem, node_points, element_bodies,
                                                   max_time=max_time - elapsed_time(start_time))
        remove_all_debug()
        if layer_plan is None:
            return None

        if post_process:
            print(SEPARATOR)
            # Allows the planner to continue to check collisions
            problem.init[:] = certificate.all_facts
            #static_facts = extract_static_facts(layer_plan, ...)
            #problem.init.extend(('Order',) + pair for pair in compute_total_orders(layer_plan))
            for fact in [('print',), ('move',)]:
                if fact in problem.init:
                    problem.init.remove(fact)
            new_layer_plan, _ = solve_pddlstream(problem, node_points, element_bodies,
                                                 planner=POSTPROCESS_PLANNER,
                                                 max_time=max_time - elapsed_time(start_time))
            if (new_layer_plan is not None) and (compute_duration(new_layer_plan) < compute_duration(layer_plan)):
                layer_plan = new_layer_plan
            user_input('{:.3f}->{:.3f}'.format(compute_duration(layer_plan), compute_duration(new_layer_plan)))

        # TODO: replan in a cost sensitive way
        layer_plan = apply_start(layer_plan, makespan)
        duration = compute_duration(layer_plan)
        makespan += duration
        print('\nLength: {} | Start: {:.3f} | End: {:.3f} | Duration: {:.3f} | Makespan: {:.3f}'.format(
            len(layer_plan), compute_start(layer_plan), compute_end(layer_plan), duration, makespan))
        full_plan.extend(layer_plan)
        removed.update(remaining)
    print(SEPARATOR)
    print_plan(full_plan)
    return full_plan, None

##################################################

def solve_stripstream(robot1, obstacles, node_points, element_bodies, ground_nodes,
                      dual=False, serialize=False, hierarchy=False, **kwargs):
    robots = mirror_robot(robot1, node_points) if dual else [robot1]
    elements = set(element_bodies)
    initial_confs = {ROBOT_TEMPLATE.format(i): Conf(robot) for i, robot in enumerate(robots)}
    saver = WorldSaver()

    layer_from_n = compute_layer_from_vertex(elements, node_points, ground_nodes)
    #layer_from_n = cluster_vertices(elements, node_points, ground_nodes) # TODO: increase resolution for small structures
    # TODO: compute directions from first, layer from second
    max_layer = max(layer_from_n.values())
    print('Max layer: {}'.format(max_layer))

    data = {}
    if serialize:
        plan, certificate = solve_serialized(robots, obstacles, node_points, element_bodies,
                                             ground_nodes, layer_from_n, initial_confs=initial_confs, **kwargs)
    else:
        plan, certificate = solve_joint(robots, obstacles, node_points, element_bodies,
                                        ground_nodes, layer_from_n, initial_confs=initial_confs, **kwargs)
    if plan is None:
        return None, data

    if hierarchy:
        print(SEPARATOR)
        static_facts = extract_static_facts(plan, certificate, initial_confs)
        partial_orders = compute_total_orders(plan)
        plan, certificate = solve_joint(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n,
                                        initial_confs=initial_confs, can_print=False, can_transit=True,
                                        additional_init=static_facts, additional_orders=partial_orders, **kwargs)
        if plan is None:
            return None, data

    if plan and not isinstance(plan[0], DurativeAction):
        time_from_start = 0.
        retimed_plan = []
        for name, args in plan:
            command = args[-1]
            command.retime(start_time=time_from_start)
            retimed_plan.append(DurativeAction(name, args, time_from_start, command.duration))
            time_from_start += command.duration
        plan = retimed_plan
    plan = reverse_plan(plan)
    print('\nLength: {} | Makespan: {:.3f}'.format(len(plan), compute_duration(plan)))
    # TODO: retime using the TFD duration
    # TODO: attempt to resolve once without any optimistic facts to see if a solution exists
    # TODO: choose a better initial config
    # TODO: decompose into layers hierarchically

    #planned_elements = [args[2] for name, args, _, _ in sorted(plan, key=lambda a: get_end(a))] # TODO: remove approach
    #if not check_plan(extrusion_path, planned_elements):
    #    return None, data

    #trajectories = None
    #trajectories = extract_trajectories(plan)
    commands = [action.args[-1] for action in reversed(plan) if action.name == 'print']
    trajectories = flatten_commands(commands)

    if has_gui():
        saver.restore()
        #label_nodes(node_points)
        # draw_ordered(recover_sequence(trajectories), node_points)
        # wait_if_gui('Continue?')

        #simulate_printing(node_points, trajectories)
        #display_trajectories(node_points, ground_nodes, trajectories)
        simulate_parallel(robots, plan)

    assert not dual
    return trajectories, data
