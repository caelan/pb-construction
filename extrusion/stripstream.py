import numpy as np

from extrusion.utils import element_supports, Profiler, load_robot
from extrusion.stream import get_print_gen_fn, USE_CONMECH
from extrusion.heuristics import compute_distance_from_node
from extrusion.visualization import draw_model, display_trajectories
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.downward import SEARCH_OPTIONS, set_cost_scale
from pddlstream.language.constants import And, PDDLProblem, print_solution
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.utils import read, get_file_path
from pybullet_tools.utils import get_configuration, wait_if_gui, RED, get_point, set_pose, Pose, Euler, Point, \
    get_movable_joints, set_joint_position, has_gui, WorldSaver


STRIPSTREAM_ALGORITHM = 'stripstream'


##################################################

def get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes,
                   trajectories=[], **kwargs):
    # TODO: instantiation slowness is due to conditional effects
    # TODO: plan for the end-effector first
    # TODO: cost-sensitive planning (greedily explore cheapest)

    # TODO: partially order the elements instead
    # TODO: full layer partial ordering
    #supports = {(e, n) for e in element_bodies for n in e if element_supports(e, n, node_points)}
    node_from_n = compute_distance_from_node(element_bodies, node_points, ground_nodes)
    supports = {(node.edge, n) for n, node in node_from_n.items() if node.edge is not None}
    #supports = set()
    # TODO: pass into the stream

    #print(supports)
    # elements = {e for e, _ in supports}
    # draw_model(elements, node_points, ground_nodes, color=RED)
    # wait_if_gui()

    initial_confs = {'r{}'.format(i): np.array(get_configuration(robot)) for i, robot in enumerate(robots)}

    domain_pddl = read(get_file_path(__file__, 'pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    constant_map = {}

    # TODO: condition on plan/downstream constraints
    # TODO: stream fusion
    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        #'sample-print': from_gen_fn(get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes)),
        'sample-print': get_wild_print_gen_fn(robots, obstacles, node_points, element_bodies, ground_nodes, **kwargs),
        'test-stiffness': from_test(test_stiffness),
        'test-cfree-traj-conf': from_test(lambda *args: True),
    }

    init = []
    for robot, conf in initial_confs.items():
        init.extend([
            ('Robot', robot),
            ('Conf', robot, conf),
            ('AtConf', robot, conf),
            #('CanMove', robot),
        ])

    init.extend(('Grounded', n) for n in ground_nodes)
    init.extend(('Supports', e, n) for e, n in supports)
    init.extend(('StartNode', n, e) for e in element_bodies
                for n in e if (e, n) not in supports)

    for e in element_bodies:
        n1, n2 = e
        #n1, n2 = ['n{}'.format(i) for i in e]
        init.extend([
            ('Node', n1),
            ('Node', n2),
            ('Element', e),
            ('Printed', e),
            ('Edge', n1, e, n2),
            ('Edge', n2, e, n1),
        ])

    assert not trajectories
    for t in trajectories:
        init.extend([
            ('Traj', t),
            ('PrintAction', t.n1, t.element, t),
        ])

    goal_literals = [
        #('AtConf', robot, initial_conf),
    ]
    #goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Removed', e) for e in element_bodies)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def plan_sequence(robot1, obstacles, node_points, element_bodies, ground_nodes,
                  trajectories=[], collisions=True, disable=False, max_time=30, checker=None):
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

    centroid = np.average(node_points, axis=0)
    #print(centroid)
    #print(get_point(robot1))
    robot2 = load_robot()
    set_pose(robot2, Pose(point=Point(*2*centroid[:2]), euler=Euler(yaw=np.pi)))

    #robots = [robot1]
    robots = [robot1, robot2]
    for robot in [robot1, robot2]:
        joint1 = get_movable_joints(robot)[0]
        set_joint_position(robot, joint1, np.pi/8)
    saver = WorldSaver()

    pddlstream_problem = get_pddlstream(robots, obstacles, node_points, element_bodies, ground_nodes,
                                        trajectories=trajectories, collisions=collisions, disable=disable,
                                        precompute_collisions=True, supports=False)
    #solution = solve_incremental(pddlstream_problem, planner='add-random-lazy', max_time=600,
    #                             max_planner_time=300, debug=True)
    stream_info = {
        'sample-print': StreamInfo(PartialInputs(unique=True)),
        'test-cfree-traj-conf': StreamInfo(p_success=1e-2, negate=True), #, verbose=False),
    }

    # TODO: goal serialization
    #set_cost_scale(1000)
    #planner = 'ff-ehc'
    #planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    #planner = 'ff-eager-tiebreak' # Need to use a eager search, otherwise doesn't incorporate new cost
    planner = 'max-astar'
    # TODO: limit the branching factor if necessary
    solution = solve_focused(pddlstream_problem, stream_info=stream_info, max_time=max_time,
                             effort_weight=1, unit_efforts=True, max_skeletons=None, unit_costs=True, bind=False,
                             planner=planner, max_planner_time=15, debug=False, reorder=False,
                             initial_complexity=1)
    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    print_solution(solution)

    plan, _, _ = solution
    data = {}
    if plan is None:
        return None, data

    trajectories = [t for name, args in reversed(plan) if name == 'print' for t in args[-1].trajectories]
    if has_gui():
        saver.restore()
        display_trajectories(node_points, ground_nodes, trajectories)
    return None, data
    #return trajectories, data

##################################################

def get_wild_print_gen_fn(robots, obstacles, node_points, element_bodies, ground_nodes,
                          collisions=True, **kwargs):
    # TODO: could reuse end-effector trajectories
    gen_fn_from_robot = {robot: get_print_gen_fn(robot, obstacles, node_points, element_bodies,
                                                 ground_nodes, **kwargs) for robot in robots}
    def wild_gen_fn(name, node1, element):
        index = int(name[1:])
        robot = robots[index]
        for command, in gen_fn_from_robot[robot](node1, element):
            q1 = np.array(command.start_conf)
            q2 = np.array(command.end_conf)
            outputs = [(q1, q2, command)]
            facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
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