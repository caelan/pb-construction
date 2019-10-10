import numpy as np
import random

from pybullet_tools.utils import get_movable_joints, get_joint_positions, multiply, invert, \
    set_joint_positions, inverse_kinematics, get_link_pose, get_distance, point_from_pose, wrap_angle, get_sample_fn, \
    link_from_name, get_pose, get_collision_fn, dump_body, get_link_subtree, wait_for_user, clone_body, \
    get_all_links, set_color, set_pose, pairwise_collision, get_relative_pose, Pose, Euler, Point, interval_generator
from extrusion.utils import TOOL_NAME, get_disabled_collisions, get_node_neighbors, \
    PrintTrajectory, retrace_supporters, get_supported_orders, prune_dominated, Command
#from extrusion.run import USE_IKFAST, get_supported_orders, retrace_supporters, SELF_COLLISIONS, USE_CONMECH
from pddlstream.language.stream import WildOutput
from pddlstream.utils import neighbors_from_orders, irange, user_input, INF

try:
    from conrob_pybullet.utils.ikfast.kuka_kr6_r900.ik import sample_tool_ik
except ImportError as e:
    print('\x1b[6;30;43m' + '{}, Using pybullet ik fn instead'.format(e) + '\x1b[0m')
    USE_IKFAST = False
    user_input("Press Enter to continue...")
else:
    USE_IKFAST = True

try:
    import pyconmech
except ImportError as e:
    print('\x1b[6;30;43m' + '{}, Not using conmech'.format(e) + '\x1b[0m')
    USE_CONMECH = False
    user_input("Press Enter to continue...")
else:
    USE_CONMECH = True

SELF_COLLISIONS = True
TOOL_ROOT = 'eef_base_link' # robot_tool0

STEP_SIZE = 0.0025  # 0.005
# 50 doesn't seem to be enough
MAX_ATTEMPTS = 1000  # 150 | 300

##################################################

#def get_grasp_rotation(direction, angle):
    #return Pose(euler=Euler(roll=np.pi / 2, pitch=direction, yaw=angle))
    #rot = Pose(euler=Euler(roll=np.pi / 2))
    #thing = (unit_point(), quat_from_vector_angle(direction, angle))
    #return multiply(thing, rot)


def get_direction_generator():
    lower = [-np.pi/2, -np.pi/2]
    upper = [+np.pi/2, +np.pi/2]
    for [roll, pitch] in interval_generator(lower, upper, use_halton=True):
        ##roll = random.uniform(0, np.pi)
        #roll = np.pi/4
        #pitch = random.uniform(0, 2*np.pi)
        #return Pose(euler=Euler(roll=np.pi / 2 + roll, pitch=pitch))
        #roll = random.uniform(-np.pi/2, np.pi/2)
        #pitch = random.uniform(-np.pi/2, np.pi/2)
        pose = Pose(euler=Euler(roll=roll, pitch=pitch))
        yield pose


def get_grasp_pose(translation, direction, angle, reverse, offset=1e-3):
    #direction = Pose(euler=Euler(roll=np.pi / 2, pitch=direction))
    return multiply(Pose(point=Point(z=offset)),
                    Pose(euler=Euler(yaw=angle)),
                    direction,
                    Pose(point=Point(z=translation)),
                    Pose(euler=Euler(roll=(1-reverse) * np.pi)))

##################################################

def compute_tool_path(element_pose, translation_path, direction, angle, reverse):
    tool_path = []
    for translation in translation_path:
        grasp_pose = get_grasp_pose(translation, direction, angle, reverse)
        tool_path.append(multiply(element_pose, invert(grasp_pose)))
    return tool_path

def tool_path_collision(tool, tool_from_root, element_pose, translation_path, direction, angle, reverse, obstacles):
    # TODO: allow sampling in the full sphere by checking collision with an element while sliding
    for tool_pose in compute_tool_path(element_pose, translation_path, direction, angle, reverse):
        set_pose(tool, multiply(tool_pose, tool_from_root))
        if any(pairwise_collision(tool, obst) for obst in obstacles):
            # TODO: sort by angle with smallest violation
            return True
    return False

def command_collision(tool_body, tool_from_root, command, bodies):
    # TODO: each new addition makes collision checking more expensive
    #offset = 4
    #for robot_conf in trajectory[offset:-offset]:
    collisions = [False for _ in range(len(bodies))]

    # TODO: separate into another method. Sort paths by tool poses first
    for trajectory in command.trajectories:
        indices = list(range(len(trajectory.path)))
        random.shuffle(indices)  # TODO: bisect
        for k in indices:
            tool_pose = trajectory.tool_path[k]
            set_pose(tool_body, multiply(tool_pose, tool_from_root))
            for i, body in enumerate(bodies):
                if not collisions[i]:
                    collisions[i] |= pairwise_collision(tool_body, body)
        for k in indices:
            robot_conf = trajectory.path[k]
            set_joint_positions(trajectory.robot, trajectory.joints, robot_conf)
            for i, body in enumerate(bodies):
                if not collisions[i]:
                    collisions[i] |= pairwise_collision(trajectory.robot, body)
    return collisions

##################################################

def optimize_angle(robot, tool, tool_from_root, tool_link, element_pose,
                   translation, direction, reverse, candidate_angles,
                   collision_fn, nearby=True, max_error=1e-2):

    movable_joints = get_movable_joints(robot)
    best_error, best_angle, best_conf = max_error, None, None
    initial_conf = get_joint_positions(robot, movable_joints)
    for angle in candidate_angles:
        grasp_pose = get_grasp_pose(translation, direction, angle, reverse)
        # Pose_{world,EE} = Pose_{world,element} * Pose_{element,EE}
        #                 = Pose_{world,element} * (Pose_{EE,element})^{-1}
        target_pose = multiply(element_pose, invert(grasp_pose))
        set_pose(tool, multiply(target_pose, tool_from_root))

        if nearby:
            set_joint_positions(robot, movable_joints, initial_conf)
        if USE_IKFAST:
            conf = sample_tool_ik(robot, target_pose, closest_only=nearby)
        else:
            if not nearby:
                # randomly sample and set joint conf for the pybullet ik fn
                sample_fn = get_sample_fn(robot, movable_joints)
                set_joint_positions(robot, movable_joints, sample_fn())
            # note that the conf get assigned inside this ik fn right away!
            conf = inverse_kinematics(robot, tool_link, target_pose)
        if (conf is None) or collision_fn(conf):
            continue

        set_joint_positions(robot, movable_joints, conf)
        link_pose = get_link_pose(robot, tool_link)
        error = get_distance(point_from_pose(target_pose), point_from_pose(link_pose))
        if error < best_error: # TODO: error a function of direction as well
            best_error, best_angle, best_conf = error, angle, conf
            #break
    if best_conf is not None:
        set_joint_positions(robot, movable_joints, best_conf)
    return best_angle, best_conf

##################################################

def compute_direction_path(robot, tool, tool_from_root,
                           length, reverse, element_bodies, element, direction,
                           obstacles, collision_fn, num_angles=1):
    """
    :param robot:
    :param length: element's length
    :param reverse: True if element end id tuple needs to be reversed
    :param element: the considered element's pybullet body
    :param direction: a sampled Pose (v \in unit sphere)
    :param collision_fn: collision checker (pybullet_tools.utils.get_collision_fn)
    note that all the static objs + elements in the support set of the considered element
    are accounted in the collision fn
    :return: feasible PrintTrajectory if found, None otherwise
    """
    #angle_step_size = np.math.radians(0.25) # np.pi / 128
    #angle_deltas = [-angle_step_size, 0, angle_step_size]
    angle_deltas = [0]
    translation_path = np.append(np.arange(-length / 2, length / 2, STEP_SIZE), [length / 2])
    element_pose = get_pose(element_bodies[element])

    #initial_angles = [wrap_angle(angle) for angle in np.linspace(0, 2*np.pi, num_angles, endpoint=False)]
    initial_angles = list(map(wrap_angle, np.random.uniform(0, 2*np.pi, num_angles))) # TODO: halton
    initial_angles = [angle for angle in initial_angles if not tool_path_collision(
        tool, tool_from_root, element_pose, translation_path, direction, angle, reverse, obstacles)]

    tool_link = link_from_name(robot, TOOL_NAME)
    initial_angle, current_conf = optimize_angle(robot, tool, tool_from_root, tool_link, element_pose,
                                                 translation_path[0], direction, reverse, initial_angles,
                                                 collision_fn, nearby=False)
    if current_conf is None:
        return None
    # TODO: constrain maximum conf displacement
    # TODO: alternating minimization for just position and also orientation
    current_angle = initial_angle
    robot_path = [current_conf]
    for translation in translation_path[1:]:
        #set_joint_positions(robot, movable_joints, current_conf)
        candidate_angles = [wrap_angle(current_angle + delta) for delta in angle_deltas]
        random.shuffle(candidate_angles)
        current_angle, current_conf = optimize_angle(robot, tool, tool_from_root, tool_link, element_pose,
                                                     translation, direction, reverse, candidate_angles,
                                                     collision_fn, nearby=True)
        if current_conf is None:
            return None
        robot_path.append(current_conf)

    tool_path = compute_tool_path(element_pose, translation_path, direction, initial_angle, reverse)
    print_traj = PrintTrajectory(robot, get_movable_joints(robot), robot_path, tool_path, element, reverse)
    return Command([print_traj])

##################################################

def get_print_gen_fn(robot, fixed_obstacles, node_points, element_bodies, ground_nodes,
                     precompute_collisions=True, supports=True, bidirectional=False,
                     collisions=True, disable=False,
                     max_directions=MAX_ATTEMPTS, max_attempts=1):
    # TODO: print on full sphere and just check for collisions with the printed element
    # TODO: can slide a component of the element down
    # TODO: prioritize choices that don't collide with too many edges

    movable_joints = get_movable_joints(robot)
    disabled_collisions = get_disabled_collisions(robot)
    #element_neighbors = get_element_neighbors(element_bodies)
    node_neighbors = get_node_neighbors(element_bodies)
    incoming_supporters, _ = neighbors_from_orders(get_supported_orders(element_bodies, node_points))

    #dump_body(robot)
    root_link = link_from_name(robot, TOOL_ROOT)
    tool_links = get_link_subtree(robot, root_link)
    tool_body = clone_body(robot, links=tool_links, visual=False, collision=True)
    #for link in get_all_links(tool_body):
    #    set_color(tool_body, np.zeros(4), link)

    tool_link = link_from_name(robot, TOOL_NAME)
    tool_from_root = get_relative_pose(robot, root_link, tool_link)

    def gen_fn(node1, element, extruded=[]): # fluents=[]):
        reverse = (node1 != element[0])
        if disable:
            path, tool_path = [], []
            traj = PrintTrajectory(robot, get_movable_joints(robot), path, tool_path, element, reverse)
            command = Command([traj])
            yield (command,)
            return

        n1, n2 = reversed(element) if reverse else element
        delta = node_points[n2] - node_points[n1]
        # if delta[2] < 0:
        #    continue
        length = np.linalg.norm(delta)  # 5cm

        #supporters = {e for e in node_neighbors[n1] if element_supports(e, n1, node_points)}
        supporters = []
        if supports:
            retrace_supporters(element, incoming_supporters, supporters)
        obstacles = set(fixed_obstacles + [element_bodies[e] for e in supporters + list(extruded)])
        if not collisions:
            obstacles = set()

        elements_order = [e for e in element_bodies if (e != element) and (element_bodies[e] not in obstacles)]
        collision_fn = get_collision_fn(robot, movable_joints, obstacles,
                                        attachments=[], self_collisions=SELF_COLLISIONS,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={}) # TODO: get_custom_limits

        direction_generator = get_direction_generator()
        trajectories = []
        for num in irange(INF):
            for attempt in irange(max_directions):
                direction = next(direction_generator)
                for _ in range(max_attempts):
                    if bidirectional:
                        reverse = random.choice([False, True])
                    command = compute_direction_path(robot, tool_body, tool_from_root,
                                                     length, reverse, element_bodies, element,
                                                     direction, obstacles, collision_fn)
                    if command is None:
                        continue
                    if precompute_collisions:
                        bodies_order = [element_bodies[e] for e in elements_order]
                        colliding = command_collision(tool_body, tool_from_root, command, bodies_order)
                        command.colliding = {e for k, e in enumerate(elements_order) if colliding[k]}
                    if (node_neighbors[n1] <= command.colliding) and not any(n in ground_nodes for n in element):
                        continue
                    trajectories.append(command)
                    prune_dominated(trajectories)
                    if command not in trajectories:
                        continue
                    print('{}) {}->{} | Supporters: {} | Attempts: {} | Trajectories: {} | Colliding: {}'.format(
                        num, n1, n2, len(supporters), attempt, len(trajectories),
                        sorted(len(t.colliding) for t in trajectories)))
                    yield (command,)
                    if precompute_collisions and not command.colliding:
                        print('Reevaluated already non-colliding trajectory!')
                        return
                    break
            else:
                print('{}) {}->{} | Supporters: {} | Attempts: {} | Max attempts exceeded!'.format(
                    num, n1, n2, len(supporters), max_directions))
                return
                #yield None
    return gen_fn

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
    import pyconmech
    elements = {fact[1] for fact in fluents}
    #print(elements)
    return True
