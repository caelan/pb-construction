import cProfile
import pstats

from extrusion.utils import element_supports, is_start_node
from extrusion.stream import get_wild_print_gen_fn, test_stiffness
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import And, PDDLProblem, print_solution
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.utils import read, get_file_path


STRIPSTREAM_ALGORITHM = 'stripstream'

def get_pddlstream(robot, obstacles, node_points, element_bodies, ground_nodes,
                   trajectories=[], **kwargs):
    # TODO: instantiation slowness is due to condition effects
    # Regression works well here because of the fixed goal state
    # TODO: plan for the end-effector first

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    constant_map = {}

    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-print': get_wild_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes, **kwargs),
        'test-stiffness': from_test(test_stiffness),
    }

    # TODO: assert that all elements have some support
    init = []
    for n in ground_nodes:
        init.append(('Grounded', n))

    nodes = set()
    for e in element_bodies:
        for n in e:
            if element_supports(e, n, node_points):
                init.append(('Supports', e, n))
            if is_start_node(n, e, node_points):
                init.append(('StartNode', n, e))
        #if e[0] not in nodes:
        #    add_text(e[0], position=(0, 0, -0.02), parent=element_bodies[e])
        #if e[1] not in nodes:
        #    add_text(e[1], position=(0, 0, 0.02), parent=element_bodies[e])
        #nodes.update(e)

    # Really there are 3 types of elements with respect to a node
    # 1) Elements that directly support the node (are below)
    # 2) Elements that stabilize the node (are parallel to the ground)
    # 3) Elements that extend from the node
    # 1 < 2 < 3

    #for n, neighbors in get_node_neighbors(element_bodies).items():
    #    for e1, e2 in permutations(neighbors, 2):
    #        p1 = node_points[get_other_node(n, e1)]
    #        p2 = node_points[get_other_node(n, e2)]
    #        if p1[2] - p2[2] > 0.01:
    #            init.append(('Above', e1, e2))

    for e in element_bodies:
        n1, n2 = e
        init.extend([
            ('Node', n1),
            ('Node', n2),
            ('Element', e),
            ('Printed', e),
            ('Edge', n1, e, n2),
            ('Edge', n2, e, n1),
            #('StartNode', n1, e),
            #('StartNode', n2, e),
        ])
        #if is_ground(e, ground_nodes):
        #    init.append(('Grounded', e))
    #for e1, neighbors in get_element_neighbors(element_bodies).items():
    #    for e2 in neighbors:
    #        init.append(('Supports', e1, e2))
    for t in trajectories:
        init.extend([
            ('Traj', t),
            ('PrintAction', t.n1, t.element, t),
        ])

    goal = And(*[('Removed', e) for e in element_bodies])

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                  trajectories=[], collisions=True, disable=False,
                  debug=False, max_time=30):
    if trajectories is None:
        return None
    # TODO: iterated search using random restarts
    # TODO: most of the time seems to be spent extracting the stream plan
    # TODO: NEGATIVE_SUFFIX to make axioms easier
    pr = cProfile.Profile()
    pr.enable()
    pddlstream_problem = get_pddlstream(robot, obstacles, node_points, element_bodies, ground_nodes,
                                        trajectories=trajectories, collisions=collisions, disable=disable)
    #solution = solve_incremental(pddlstream_problem, planner='add-random-lazy', max_time=600,
    #                             max_planner_time=300, debug=True)
    stream_info = {
        'sample-print': StreamInfo(PartialInputs(unique=True)),
    }
    # TODO: goal serialization
    #planner = 'ff-ehc'
    #planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    planner = 'ff-eager-tiebreak' # Need to use a eager search, otherwise doesn't incorporate new cost
    #planner = 'max-astar'
    # TODO: limit the branching factor if necessary
    solution = solve_focused(pddlstream_problem, stream_info=stream_info, max_time=max_time,
                             effort_weight=1, unit_efforts=True, max_skeletons=None, unit_costs=True, bind=False,
                             planner=planner, max_planner_time=15, debug=debug, reorder=False)
    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    print_solution(solution)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(25)
    plan, _, _ = solution
    data = {}
    if plan is None:
        return None, data
    trajectories = [t for _, (n1, e, c) in reversed(plan)
            for t in c.trajectories]
    return trajectories, data
