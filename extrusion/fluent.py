from extrusion.stiffness import USE_CONMECH, test_stiffness
from extrusion.stream import get_print_gen_fn
from extrusion.temporal import index_from_name
from extrusion.utils import TOOL_LINK, compute_printed_nodes, check_connected, compute_z_distance
from pddlstream.algorithms.downward import parse_action
from pddlstream.language.constants import get_prefix, get_args
from pddlstream.language.conversion import obj_from_pddl
from pybullet_tools.utils import point_from_pose, get_link_pose, link_from_name, get_distance

# TODO: use stream statistics for ordering

def get_location_distance(node_points, robots=[], initial_confs={}):

    def extract_point(loc):
        if isinstance(loc, str):
            name = loc.split('-')[0]
            conf = initial_confs[name]
            conf.assign()
            robot = index_from_name(robots, name)
            return point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK)))
        else:
            return node_points[loc]

    def fn(*locations):
        return 1. + get_distance(*map(extract_point, locations))
    return fn


def extract_printed(fluents):
    assert all(get_prefix(fact) == 'printed' for fact in fluents)
    return {get_args(fact)[0] for fact in fluents}


def get_test_printable(ground_nodes, debug=True):
    def test_printable(node1, element, fluents=[]):
        printed = extract_printed(fluents)
        next_printed = printed - {element}
        if debug:
            print(test_printable.__name__, node1, element, len(next_printed), next_printed)
            #user_input()
        # TODO: should be connected before and after the extrusion
        # Building from connected node and connected structure
        next_nodes = compute_printed_nodes(ground_nodes, next_printed)
        return (node1 in next_nodes) and check_connected(ground_nodes, next_printed)
    return test_printable


def get_test_stiff(debug=True):
    def test_stiff(fluents=[]):
        printed = extract_printed(fluents)
        if debug:
            print(test_stiff.__name__, len(printed), printed)
            #user_input()
        if not USE_CONMECH:
           return True
        # https://github.com/yijiangh/conmech
        # TODO: to use the non-skeleton focused algorithm, need to remove the negative axiom upon success
        return True
    return test_stiff


def get_fluent_print_gen_fn(robots, static_obstacles, node_points, element_bodies, ground_nodes,
                            connectivity=False, stiffness=False, debug=True, **kwargs):
    #wild_print_gen_fn = get_wild_print_gen_fn(robots, static_obstacles, node_points, element_bodies, ground_nodes,
    #                                          initial_confs={}, return_home=False, **kwargs) # collisions=False,
    print_gen_fn_from_robot = {robot: get_print_gen_fn(robot, static_obstacles, node_points, element_bodies, ground_nodes,
                                                       precompute_collisions=False, **kwargs) for robot in robots}

    test_printable = get_test_printable(ground_nodes, debug=debug)
    #test_stiff = get_test_stiff(debug=debug)

    def gen_fn(name, node1, element, node2, fluents=[]):
        robot = index_from_name(robots, name)
        printed = extract_printed(fluents)
        next_printed = printed - {element}
        if connectivity and not test_printable(node1, element, fluents=fluents):
            return
        if stiffness and not test_stiffness(extrusion_path, element_from_id, next_printed, checker=checker):
            return
        generator = print_gen_fn_from_robot[robot](node1, element, extruded=next_printed)
        #generator = islice(generator, stop=1)
        #return generator
        for print_cmd, in generator:
            yield (print_cmd,)
            #break
    return gen_fn


def get_order_fn(node_points):
    # TODO: general heuristic function
    # TODO: bias toward nearby elements
    #heuristic_fn = get_heuristic_fn(robot=None, extrusion_path=None, heuristic='z', checker=None, forward=False)

    def order_fn(state, goal, operators):
        from strips.utils import ha_applicable
        actions = ha_applicable(state, goal, operators) # filters axioms
        action_priorities = {}
        for action in actions:
            name, args = parse_action(action.fd_action.name)
            args = [obj_from_pddl(arg).value for arg in args]
            if name == 'print':
                _, _, element, _, _ = args
                priority = -compute_z_distance(node_points, element)
            elif name == 'move':
                _, loc1, loc2 = args
                priority = 0. # TODO: use the location of the best element here
            else:
                raise NotImplementedError(name)
            action_priorities[action] = priority
        return sorted(actions, key=action_priorities.__getitem__)
    return order_fn