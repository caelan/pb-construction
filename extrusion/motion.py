from __future__ import print_function

import time
import numpy as np
import colorsys
import random
from termcolor import cprint

from scipy.spatial.qhull import QhullError
from collections import Counter

from pybullet_tools.utils import get_movable_joints, set_joint_positions, plan_joint_motion, \
    connect, point_from_pose, get_link_pose, link_from_name, add_line, \
    wait_for_duration, disconnect, elapsed_time, reset_simulation, wait_for_user, convex_hull, \
    create_mesh, draw_mesh, apply_alpha, RED, remove_body, pairwise_collision, randomize, \
    get_sample_fn, get_distance_fn, get_extend_fn, get_collision_fn, \
    check_initial_end, birrt, INF, get_bodies_in_region, get_aabb, spaced_colors, vertices_from_data, \
    BASE_LINK, vertices_from_link, apply_affine, get_pose

from extrusion.utils import get_disabled_collisions, MotionTrajectory, load_world, PrintTrajectory, is_ground, \
    RESOLUTION, JOINT_WEIGHTS
from extrusion.visualization import draw_ordered, set_extrusion_camera
from extrusion.stream import SELF_COLLISIONS

MIN_ELEMENTS = 3 # 2 | INF

def create_bounding_mesh(element_bodies, node_points, printed_elements):
    # TODO: use bounding boxes instead of points
    # TODO: connected components
    #printed_points = [node_points[n] for element in printed_elements for n in element]
    printed_points = []
    for element in printed_elements:
        body = element_bodies[element]
        printed_points.extend(apply_affine(get_pose(body), vertices_from_link(body, BASE_LINK)))

    rgb = colorsys.hsv_to_rgb(h=random.random(), s=1, v=1)
    #rgb = RED
    try:
        mesh = convex_hull(printed_points)
        # handles = draw_mesh(mesh)
        return create_mesh(mesh, under=True, color=apply_alpha(rgb, 0.5))
        # TODO: check collisions with hull before the elements
    except QhullError as e:
        print(printed_elements)
        raise e
        #return None

def compute_motion(robot, fixed_obstacles, element_bodies, node_points,
                   printed_elements, start_conf, end_conf, collisions=True):
    # TODO: can also just plan to initial conf and then shortcut
    joints = get_movable_joints(robot)
    assert len(joints) == len(end_conf)
    weights = JOINT_WEIGHTS
    resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
    disabled_collisions = get_disabled_collisions(robot)
    custom_limits = {}
    #element_from_body = {b: e for e, b in element_bodies.items()}

    frequencies = {}
    for element in printed_elements:
        z = np.average([node_points[n][2] for n in element])
        #key = np.round(2*z, 1)
        key = None
        frequencies.setdefault(key, []).append(element)
    print(Counter({key: len(elements) for key, elements in frequencies.items()}))

    # TODO: apply this elsewhere
    obstacles = list(fixed_obstacles)
    hulls = {}
    for elements in frequencies.values():
        element_obstacles = randomize(element_bodies[e] for e in elements)
        if MIN_ELEMENTS <= len(elements):
            hull = create_bounding_mesh(element_bodies, node_points, elements)
            assert hull is not None
            hulls[hull] = element_obstacles
        else:
            obstacles.extend(element_obstacles)

    if not collisions:
        obstacles = []
        hulls = {}
    #print(hulls)
    #print(obstacles)
    #wait_for_user()

    sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(robot, joints, weights=weights)
    extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(robot, joints, obstacles, attachments={}, self_collisions=SELF_COLLISIONS,
                                    disabled_collisions=disabled_collisions, custom_limits=custom_limits, max_distance=0.)

    def element_collision_fn(q):
        if collision_fn(q):
            return True
        #for body in get_bodies_in_region(get_aabb(robot)): # Perform per link?
        #    if (element_from_body.get(body, None) in printed_elements) and pairwise_collision(robot, body):
        #        return True
        for hull, bodies in hulls.items():
            if pairwise_collision(robot, hull) and any(pairwise_collision(robot, body) for body in bodies):
                return True
        return False

    path = None
    if check_initial_end(start_conf, end_conf, collision_fn):
        path = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, element_collision_fn,
                     restarts=50, iterations=100, smooth=100)

    # path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles,
    #                          self_collisions=SELF_COLLISIONS, disabled_collisions=disabled_collisions,
    #                          weights=weights, resolutions=resolutions,
    #                          restarts=50, iterations=100, smooth=100)

    for hull in hulls:
        remove_body(hull)
    if path is None:
        cprint('Failed to find a transition motion plan!', 'red')
        return None
    return MotionTrajectory(robot, joints, path)

def compute_motions(robot, fixed_obstacles, element_bodies, node_points, initial_conf, print_trajectories, **kwargs):
    # TODO: reoptimize for the sequence that have the smallest movements given this
    # TODO: sample trajectories
    # TODO: more appropriate distance based on displacement/volume
    if print_trajectories is None:
        return None
    if any(isinstance(traj, MotionTrajectory) for traj in print_trajectories):
        return print_trajectories
    start_time = time.time()
    printed_elements = []
    all_trajectories = []
    start_confs = [initial_conf] + [traj.path[-1] for traj in print_trajectories]
    end_confs = [traj.path[0] for traj in print_trajectories] + [initial_conf]
    for i, (start_conf, end_conf) in enumerate(zip(start_confs, end_confs)):
        motion_traj = compute_motion(robot, fixed_obstacles, element_bodies, node_points,
                                     printed_elements, start_conf, end_conf, **kwargs)
        if motion_traj is None:
            return None
        print('{}) {} | Time: {:.3f}'.format(i, motion_traj, elapsed_time(start_time)))
        all_trajectories.append(motion_traj)
        if i < len(print_trajectories):
            print_traj = print_trajectories[i]
            printed_elements.append(print_traj.element)
            all_trajectories.append(print_traj)
    return all_trajectories

##################################################

def display_trajectories(node_points, ground_nodes, trajectories, animate=True, time_step=0.02):
    if trajectories is None:
        return
    connect(use_gui=True)
    set_extrusion_camera(node_points)
    obstacles, robot = load_world()
    movable_joints = get_movable_joints(robot)
    if not animate:
        planned_elements = [traj.element for traj in trajectories]
        draw_ordered(planned_elements, node_points)
        wait_for_user()
        disconnect()
        return

    wait_for_user()
    #element_bodies = dict(zip(elements, create_elements(node_points, elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    connected_nodes = set(ground_nodes)
    print('Trajectories:', len(trajectories))
    for i, trajectory in enumerate(trajectories):
        #wait_for_user()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []
        for conf in trajectory.path:
            set_joint_positions(robot, movable_joints, conf)
            if isinstance(trajectory, PrintTrajectory):
                current_point = point_from_pose(trajectory.end_effector.get_tool_pose())
                if last_point is not None:
                    color = (0, 0, 1) if is_ground(trajectory.element, ground_nodes) else (1, 0, 0)
                    handles.append(add_line(last_point, current_point, color=color))
                last_point = current_point
            if time_step is None:
                wait_for_user()
            else:
                wait_for_duration(time_step)

        if isinstance(trajectory, PrintTrajectory):
            is_connected = (trajectory.n1 in connected_nodes) # and (trajectory.n2 in connected_nodes)
            print('{}) {:9} | Connected: {} | Ground: {} | Length: {}'.format(
                i, str(trajectory), is_connected, is_ground(trajectory.element, ground_nodes), len(trajectory.path)))
            if not is_connected:
                wait_for_user()
            connected_nodes.add(trajectory.n2)

    wait_for_user()
    reset_simulation()
    disconnect()
