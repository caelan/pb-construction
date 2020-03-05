from extrusion.utils import element_supports, Profiler
from extrusion.stream import get_print_gen_fn, USE_CONMECH
from extrusion.heuristics import compute_distance_from_node
from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import And, PDDLProblem, print_solution
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.utils import read, get_file_path


STRIPSTREAM_ALGORITHM = 'stripstream'


##################################################

# TODO: condition on plan/downstream constraints
# TODO: stream fusion

def compute_z_supports(node_points, element_bodies):
    return {(e, n) for e in element_bodies for n in e if element_supports(e, n, node_points)}


##################################################

def get_pddlstream(robot, obstacles, node_points, element_bodies, ground_nodes,
                   trajectories=[], **kwargs):
    # TODO: instantiation slowness is due to conditional effects
    # TODO: plan for the end-effector first
    # TODO: cost-sensitive planning (greedily explore cheapest)

    # TODO: partially order the elements instead
    # TODO: full layer partial ordering
    #supports = compute_z_supports(node_points, element_bodies)
    node_from_n = compute_distance_from_node(element_bodies, node_points, ground_nodes)
    supports = {(node.edge, n) for n, node in node_from_n.items() if node.edge is not None}
    #supports = set()
    # TODO: pass into the stream

    domain_pddl = read(get_file_path(__file__, 'pddl/retired.pddl'))
    constant_map = {}

    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-print': get_wild_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes, **kwargs),
        'test-stiffness': from_test(test_stiffness),
    }

    init = []
    init.extend(('Grounded', n) for n in ground_nodes)
    init.extend(('Supports', e, n) for e, n in supports)
    init.extend(('StartNode', n, e) for e in element_bodies
                for n in e if (e, n) not in supports)

    for e in element_bodies:
        n1, n2 = e
        init.extend([
            ('Node', n1),
            ('Node', n2),
            ('Element', e),
            ('Printed', e),
            ('Edge', n1, e, n2),
            ('Edge', n2, e, n1),
        ])
    for t in trajectories:
        init.extend([
            ('Traj', t),
            ('PrintAction', t.n1, t.element, t),
        ])

    goal_literals = []
    goal_literals.extend(('Removed', e) for e in element_bodies)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                  trajectories=[], collisions=True, disable=False, debug=False, max_time=30, checker=None):
    # TODO: fail if wild stream produces unexpected facts
    # TODO: try search at different cost levels (i.e. w/ and w/o abstract)
    # TODO: only consider axioms that could be relevant
    if trajectories is None:
        return None
    # TODO: iterated search using random restarts
    # TODO: most of the time seems to be spent extracting the stream plan
    # TODO: NEGATIVE_SUFFIX to make axioms easier
    # TODO: sort by action cost heuristic
    # http://www.fast-downward.org/Doc/Evaluator#Max_evaluator

    with Profiler():
        pddlstream_problem = get_pddlstream(robot, obstacles, node_points, element_bodies, ground_nodes,
                                            trajectories=trajectories, collisions=collisions, disable=disable,
                                            precompute_collisions=True, supports=False)
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

    plan, _, _ = solution
    data = {}
    if plan is None:
        return None, data
    trajectories = [t for _, (n1, e, c) in reversed(plan) for t in c.trajectories]
    return trajectories, data

##################################################

def get_wild_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                          collisions=True, **kwargs):
    gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes, **kwargs)
    def wild_gen_fn(node1, element):
        for t, in gen_fn(node1, element):
            outputs = [(t,)]
            facts = [('Collision', t, e2) for e2 in t.colliding] if collisions else []
            yield WildOutput(outputs, facts)
    return wild_gen_fn


def test_stiffness(fluents=[]):
    assert all(fact[0] == 'printed' for fact in fluents)
    if not USE_CONMECH:
       return True
    # https://github.com/yijiangh/conmech
    # TODO: to use the non-skeleton focused algorithm, need to remove the negative axiom upon success
    elements = {fact[1] for fact in fluents}
    #print(elements)
    return True