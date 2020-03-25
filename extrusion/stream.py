import numpy as np
import random
import time

from itertools import cycle
from itertools import islice

from pybullet_tools.utils import get_movable_joints, get_joint_positions, multiply, invert, \
    set_joint_positions, inverse_kinematics, get_link_pose, get_distance, point_from_pose, wrap_angle, get_sample_fn, \
    link_from_name, get_pose, get_collision_fn, set_pose, pairwise_collision, Pose, Euler, Point, interval_generator, \
    randomize, get_extend_fn, user_input, INF, elapsed_time, get_bodies_in_region, get_aabb, get_all_links, \
    pairwise_link_collision, step_simulation, BASE_LINK, get_configuration, get_point, get_unit_vector, draw_pose, \
    wait_if_gui, draw_point, tform_point, set_color, GREEN, RED
from extrusion.utils import TOOL_LINK, get_disabled_collisions, get_node_neighbors, \
    PrintTrajectory, retrace_supporters, prune_dominated, Command, MotionTrajectory, RESOLUTION, \
    JOINT_WEIGHTS, EE_LINK, EndEffector, is_ground, is_end, is_reversed, reverse_element, set_configuration
from pddlstream.utils import neighbors_from_orders, irange

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
ORTHOGONAL_GROUND = False

STEP_SIZE = 1e-3  # 0.0025
APPROACH_DISTANCE = 0.01
JOINT_RESOLUTIONS = np.divide(0.25 * RESOLUTION * np.ones(JOINT_WEIGHTS.shape), JOINT_WEIGHTS)

MAX_DIRECTIONS = 500
MAX_ATTEMPTS = 1
TRANSLATION_TOLERANCE = 1e-2

# TODO: wire wrapping collision checker

##################################################

#def get_grasp_rotation(direction, angle):
    #return Pose(euler=Euler(roll=np.pi / 2, pitch=direction, yaw=angle))
    #rot = Pose(euler=Euler(roll=np.pi / 2))
    #thing = (unit_point(), quat_from_vector_angle(direction, angle))
    #return multiply(thing, rot)


def get_direction_generator(end_effector, **kwargs):
    world_from_base = get_pose(end_effector.robot)
    lower = [-np.pi/2, -np.pi/2]
    upper = [+np.pi/2, +np.pi/2]
    for [roll, pitch] in interval_generator(lower, upper, **kwargs):
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

def compute_tool_path(element_pose, translation_path, direction, angle, reverse):
    tool_path = []
    for translation in translation_path:
        grasp_pose = get_grasp_pose(translation, direction, angle, reverse)
        tool_path.append(multiply(element_pose, invert(grasp_pose)))
    return tool_path

##################################################

def tool_path_collision(end_effector, element_pose, translation_path, direction, angle, reverse, obstacles):
    # TODO: allow sampling in the full sphere by checking collision with an element while sliding
    for tool_pose in randomize(compute_tool_path(element_pose, translation_path, direction, angle, reverse)):
        end_effector.set_pose(tool_pose)
        #bodies = obstacles
        tool_aabb = get_aabb(end_effector.body) # TODO: could just translate
        #handles = draw_aabb(tool_aabb)
        bodies = {b for b, _ in get_bodies_in_region(tool_aabb) if b in obstacles}
        #print(bodies)
        #for body, link in bodies:
        #    handles.extend(draw_aabb(get_aabb(body, link)))
        #wait_for_user()
        #remove_handles(handles)
        if any(pairwise_collision(end_effector.body, obst) for obst in bodies):
            # TODO: sort by angle with smallest violation
            return True
    return False

def command_collision(end_effector, command, bodies):
    # TODO: each new addition makes collision checking more expensive
    #offset = 4
    #for robot_conf in trajectory[offset:-offset]:
    collisions = [False for _ in range(len(bodies))]
    # Orientation remains the same for the extrusion trajectory
    idx_from_body = dict(zip(bodies, range(len(bodies))))
    # TODO: use bounding cylinder for each element
    # TODO: separate into another method. Sort paths by tool poses first
    for trajectory in command.trajectories:
        for tool_pose in randomize(trajectory.get_link_path()): # TODO: bisect
            end_effector.set_pose(tool_pose)
            #for body, _ in get_bodies_in_region(tool_aabb):
            for i, body in enumerate(bodies):
                if body not in idx_from_body: # Robot
                    continue
                idx = idx_from_body[body]
                if not collisions[idx]:
                    collisions[idx] |= pairwise_collision(end_effector.body, body)
    for trajectory in command.trajectories:
        for robot_conf in randomize(trajectory.path):
            set_joint_positions(trajectory.robot, trajectory.joints, robot_conf)
            for i, body in enumerate(bodies):
                if not collisions[i]:
                    collisions[i] |= pairwise_collision(trajectory.robot, body)
    #for element, unsafe in zip(elements, collisions):
    #    command.safe_per_element[element] = unsafe
    return collisions

##################################################

def solve_ik(end_effector, target_pose, nearby=True):
    robot = end_effector.robot
    movable_joints = get_movable_joints(robot)
    if USE_IKFAST:
        # TODO: sample from the set of solutions
        conf = sample_tool_ik(robot, target_pose, closest_only=nearby, get_all=False)
    else:
        # TODO: repeat several times
        # TODO: condition on current config
        if not nearby:
            # randomly sample and set joint conf for the pybullet ik fn
            sample_fn = get_sample_fn(robot, movable_joints)
            set_joint_positions(robot, movable_joints, sample_fn())
        # note that the conf get assigned inside this ik fn right away!
        conf = inverse_kinematics(robot, end_effector.tool_link, target_pose)
    return conf

def optimize_angle(end_effector, element_pose,
                   translation, direction, reverse, candidate_angles,
                   collision_fn, nearby=True, max_error=TRANSLATION_TOLERANCE):
    robot = end_effector.robot
    movable_joints = get_movable_joints(robot)
    best_error, best_angle, best_conf = max_error, None, None
    initial_conf = get_joint_positions(robot, movable_joints)
    for angle in candidate_angles:
        grasp_pose = get_grasp_pose(translation, direction, angle, reverse)
        # Pose_{world,EE} = Pose_{world,element} * Pose_{element,EE}
        #                 = Pose_{world,element} * (Pose_{EE,element})^{-1}
        target_pose = multiply(element_pose, invert(grasp_pose))
        set_pose(end_effector.body, multiply(target_pose, end_effector.tool_from_ee))

        if nearby:
            set_joint_positions(robot, movable_joints, initial_conf)
        conf = solve_ik(end_effector, target_pose, nearby=nearby)
        if (conf is None) or collision_fn(conf):
            continue

        set_joint_positions(robot, movable_joints, conf)
        link_pose = get_link_pose(robot, end_effector.tool_link)
        error = get_distance(point_from_pose(target_pose), point_from_pose(link_pose))
        if error < best_error: # TODO: error a function of direction as well
            best_error, best_angle, best_conf = error, angle, conf
            #break
    if best_conf is not None:
        set_joint_positions(robot, movable_joints, best_conf)
    return best_angle, best_conf

##################################################

def plan_approach(end_effector, print_traj, collision_fn, approach_distance=APPROACH_DISTANCE):
    # TODO: collisions at the ends of elements
    if approach_distance == 0:
        return Command([print_traj])
    robot = end_effector.robot
    joints = get_movable_joints(robot)
    extend_fn = get_extend_fn(robot, joints, resolutions=0.25*JOINT_RESOLUTIONS)
    tool_link = link_from_name(robot, TOOL_LINK)
    approach_pose = Pose(Point(z=-approach_distance))

    #element = print_traj.element
    #midpoint = get_point(element)
    #points = map(point_from_pose, [print_traj.tool_path[0], print_traj.tool_path[-1]])
    #midpoint = np.average(list(points), axis=0)
    #draw_point(midpoint)
    #element_to_base = 0.05*get_unit_vector(get_point(robot) - midpoint)
    #retreat_pose = Pose(Point(element_to_base))
    # TODO: perpendicular to element

    # TODO: solve_ik
    start_conf = print_traj.path[0]
    set_joint_positions(robot, joints, start_conf)
    initial_pose = multiply(print_traj.tool_path[0], approach_pose)
    #draw_pose(initial_pose)
    #wait_if_gui()
    initial_conf = inverse_kinematics(robot, tool_link, initial_pose)
    if initial_conf is None:
        return None
    initial_path = [initial_conf] + list(extend_fn(initial_conf, start_conf))
    if any(map(collision_fn, initial_path)):
        return None
    initial_traj = MotionTrajectory(robot, joints, initial_path)

    end_conf = print_traj.path[-1]
    set_joint_positions(robot, joints, end_conf)
    final_pose = multiply(print_traj.tool_path[-1], approach_pose)
    final_conf = inverse_kinematics(robot, tool_link, final_pose)
    if final_conf is None:
        return None
    final_path = [end_conf] + list(extend_fn(end_conf, final_conf)) # TODO: version that includes the endpoints
    if any(map(collision_fn, final_path)):
        return None
    final_traj = MotionTrajectory(robot, joints, final_path)
    return Command([initial_traj, print_traj, final_traj])

##################################################

class ToolTrajectory(object):
    def __init__(self, extrusion, direction, angle):
        self.extrusion = extrusion
        self.direction = direction
        self.angle = angle
        self.tool_path = compute_tool_path(self.extrusion.element_pose, self.extrusion.translation_path,
                                           direction, angle, self.extrusion.reverse)
        self.safe_from_body = {}
        self.trajectories = []
    def is_safe(self, obstacles):
        checked_bodies = set(self.safe_from_body) & set(obstacles)
        if any(not self.safe_from_body[e] for e in checked_bodies):
            return False
        unchecked_bodies = randomize(set(obstacles) - checked_bodies)
        if not unchecked_bodies:
            return True
        for tool_pose in randomize(self.tool_path):
            self.extrusion.end_effector.set_pose(tool_pose)
            tool_aabb = get_aabb(self.extrusion.end_effector.body) # TODO: cache nearby_bodies
            nearby_bodies = {b for b, _ in get_bodies_in_region(tool_aabb) if b in unchecked_bodies}
            for body in nearby_bodies:
                if pairwise_collision(self.extrusion.end_effector.body, body):
                    self.safe_from_body[body] = False
                    return False
        return True
    def __iter__(self):
        return [self.extrusion, self.direction, self.angle, self.tool_path]

class Extrusion(object):
    def __init__(self, end_effector, element_bodies, node_points, ground_nodes, element, reverse, max_directions=INF):
        # TODO: more directions for ground collisions
        self.end_effector = end_effector
        self.element = element
        self.reverse = reverse
        self.max_directions = max_directions
        self.element_pose = get_pose(element_bodies[element])
        n1, n2 = element
        length = get_distance(node_points[n1], node_points[n2])
        self.translation_path = np.append(np.arange(-length / 2, length / 2, STEP_SIZE), [length / 2])
        if ORTHOGONAL_GROUND and is_ground(element, ground_nodes):
            # TODO: orthogonal to the ground
            self.direction_generator = cycle([Pose(euler=Euler(roll=0, pitch=0))])
        else:
            self.direction_generator = get_direction_generator(self.end_effector, use_halton=False)
        self.trajectories = []
    @property
    def directed(self):
        return reverse_element(self.element) if self.reverse else self.element
    def sample_direction(self):
        angle = wrap_angle(random.uniform(0, 2*np.pi)) # TODO: halton generator
        direction = next(self.direction_generator)
        trajectory = ToolTrajectory(self, direction, angle)
        self.trajectories.append(trajectory)
        return trajectory
    def generator(self):
        for direction in self.trajectories:
            yield direction
        while len(self.trajectories) < self.max_directions:
            yield self.sample_direction()

##################################################

def compute_direction_path(tool_traj, collision_fn, initial_conf=None, ee_only=False, **kwargs):
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
    direction = tool_traj.direction
    initial_angles = [tool_traj.angle]
    tool_path = tool_traj.tool_path
    extrusion = tool_traj.extrusion

    element = extrusion.element
    reverse = extrusion.reverse
    element_pose = extrusion.element_pose
    translation_path = extrusion.translation_path

    end_effector = extrusion.end_effector
    robot = end_effector.robot

    if ee_only:
        robot_path = []
        print_traj = PrintTrajectory(end_effector, get_movable_joints(robot), robot_path, tool_path, element, reverse)
        # TODO: plan_approach
        return Command([print_traj])

    max_error = TRANSLATION_TOLERANCE
    nearby = initial_conf is not None
    if nearby:
        max_error = INF
        set_configuration(robot, initial_conf)

    initial_angle, current_conf = optimize_angle(end_effector, element_pose,
                                                 translation_path[0], direction, reverse, initial_angles,
                                                 collision_fn, nearby=False, max_error=max_error)
    if current_conf is None:
        return None
    # TODO: constrain maximum conf displacement
    # TODO: alternating minimization for just position and also orientation
    #angle_step_size = np.math.radians(0.25) # np.pi / 128
    #angle_deltas = [-angle_step_size, 0, angle_step_size]
    angle_deltas = [0]

    current_angle = initial_angle
    robot_path = [current_conf]
    for translation in translation_path[1:]:
        #set_joint_positions(robot, movable_joints, current_conf)
        candidate_angles = [wrap_angle(current_angle + delta) for delta in angle_deltas]
        random.shuffle(candidate_angles)
        current_angle, current_conf = optimize_angle(end_effector, element_pose,
                                                     translation, direction, reverse, candidate_angles,
                                                     collision_fn, nearby=True)
        if current_conf is None:
            return None
        robot_path.append(current_conf)

    tool_path = compute_tool_path(element_pose, translation_path, direction, initial_angle, reverse)
    print_traj = PrintTrajectory(end_effector, get_movable_joints(robot), robot_path, tool_path, element, reverse)
    return plan_approach(end_effector, print_traj, collision_fn, **kwargs)

##################################################

def get_element_collision_fn(robot, obstacles):
    joints = get_movable_joints(robot)
    disabled_collisions = get_disabled_collisions(robot)
    custom_limits = {} # get_custom_limits(robot) # specified within the kuka URDF
    robot_links = get_all_links(robot) # Base link isn't real
    #robot_links = get_links(robot)

    collision_fn = get_collision_fn(robot, joints, obstacles=[],
                                    attachments=[], self_collisions=SELF_COLLISIONS,
                                    disabled_collisions=disabled_collisions,
                                    custom_limits=custom_limits)

    # TODO: precompute bounding boxes and manually check
    #for body in obstacles: # Calling before get_bodies_in_region does not act as step_simulation
    #    get_aabb(body, link=None)
    step_simulation() # Okay to call only once and then just ignore the robot
    # TODO: call this once globally

    def element_collision_fn(q):
        if collision_fn(q):
            return True
        #step_simulation()  # Might only need to call this once
        for robot_link in robot_links:
            #bodies = obstacles
            aabb = get_aabb(robot, link=robot_link)
            bodies = {b for b, _ in get_bodies_in_region(aabb) if b in obstacles}
            #set_joint_positions(robot, joints, q) # Need to reset
            #draw_aabb(aabb)
            #print(robot_link, get_link_name(robot, robot_link), len(bodies), aabb)
            #print(sum(pairwise_link_collision(robot, robot_link, body, link2=0) for body, _ in region_bodies))
            #print(sum(pairwise_collision(robot, body) for body, _ in region_bodies))
            #wait_for_user()
            if any(pairwise_link_collision(robot, robot_link, body, link2=BASE_LINK) for body in bodies):
                #wait_for_user()
                return True
        return False
    return element_collision_fn

##################################################

SKIP_PERCENTAGE = 0.0 # 0.0 | 0.95

def get_print_gen_fn(robot, fixed_obstacles, node_points, element_bodies, ground_nodes,
                     precompute_collisions=False, partial_orders=set(), removed=set(),
                     collisions=True, disable=False, ee_only=False, allow_failures=False,
                     p_nearby=0., max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS, max_time=INF, **kwargs):
    # TODO: print on full sphere and just check for collisions with the printed element
    # TODO: can slide a component of the element down
    if not collisions:
        precompute_collisions = False
    #element_neighbors = get_element_neighbors(element_bodies)
    node_neighbors = get_node_neighbors(element_bodies)
    incoming_supporters, _ = neighbors_from_orders(partial_orders)

    initial_conf = get_configuration(robot)
    end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK),
                               tool_link=link_from_name(robot, TOOL_LINK))
    extrusions = {reverse_element(element) if reverse else element:
                      Extrusion(end_effector, element_bodies, node_points, ground_nodes, element, reverse)
                  for element in element_bodies for reverse in [False, True]}

    def gen_fn(node1, element, extruded=[], trajectories=[]): # fluents=[]):
        start_time = time.time()
        assert not is_reversed(element_bodies, element)
        idle_time = 0
        reverse = is_end(node1, element)
        if disable or (len(extruded) < SKIP_PERCENTAGE*len(element_bodies)): # For quick visualization
            path, tool_path = [], []
            traj = PrintTrajectory(end_effector, get_movable_joints(robot), path, tool_path, element, reverse)
            command = Command([traj])
            yield (command,)
            return
        directed = reverse_element(element) if reverse else element
        extrusion = extrusions[directed]

        n1, n2 = reverse_element(element) if reverse else element
        neighboring_elements = node_neighbors[n1] & node_neighbors[n2]

        supporters = [] # TODO: can also do according to levels
        retrace_supporters(element, incoming_supporters, supporters)
        element_obstacles = {element_bodies[e] for e in supporters + list(extruded)}
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()
            #obstacles = set(fixed_obstacles)

        elements_order = [e for e in element_bodies if (e != element) and (e not in removed)
                          and (element_bodies[e] not in obstacles)]
        collision_fn = get_element_collision_fn(robot, obstacles)
        #print(len(fixed_obstacles), len(element_obstacles), len(elements_order))

        trajectories = list(trajectories)
        for num in irange(INF):
            for attempt, tool_traj in enumerate(islice(extrusion.generator(), max_directions)):
                # TODO: is this slower now for some reason?
                #tool_traj.safe_from_body = {} # Disables caching
                if not tool_traj.is_safe(obstacles):
                    continue
                for _ in range(max_attempts):
                    if max_time <= elapsed_time(start_time):
                        return
                    nearby_conf = initial_conf if random.random() < p_nearby else None
                    command = compute_direction_path(tool_traj, collision_fn, ee_only=ee_only,
                                                     initial_conf=nearby_conf, **kwargs)
                    if command is None:
                        continue

                    command.update_safe(extruded)
                    if precompute_collisions:
                        bodies_order = [element_bodies[e] for e in elements_order]
                        colliding = command_collision(end_effector, command, bodies_order)
                        for element2, unsafe in zip(elements_order, colliding):
                            if unsafe:
                                command.set_unsafe(element2)
                            else:
                                command.set_safe(element2)
                    if not is_ground(element, ground_nodes) and (neighboring_elements <= command.colliding):
                        continue # If all neighbors collide

                    trajectories.append(command)
                    if precompute_collisions:
                        prune_dominated(trajectories)
                    if command not in trajectories:
                        continue
                    print('{}) {}->{} | EE: {} | Supporters: {} | Attempts: {} | Trajectories: {} | Colliding: {}'.format(
                        num, n1, n2, ee_only, len(supporters), attempt, len(trajectories),
                        sorted(len(t.colliding) for t in trajectories)))
                    temp_time = time.time()
                    yield (command,)

                    idle_time += elapsed_time(temp_time)
                    if precompute_collisions:
                        if len(command.colliding) == 0:
                            #print('Reevaluated already non-colliding trajectory!')
                            return
                        elif len(command.colliding) == 1:
                            [colliding_element] = command.colliding
                            obstacles.add(element_bodies[colliding_element])
                    break
                else:
                    if allow_failures:
                        yield None
            else:
                print('{}) {}->{} | EE: {} | Supporters: {} | Attempts: {} | Max attempts exceeded!'.format(
                    num, n1, n2, ee_only, len(supporters), max_directions))
                return
                #yield None
    return gen_fn
