from __future__ import print_function

import time
import numpy as np

from pybullet_tools.utils import get_movable_joints, set_joint_positions, plan_joint_motion, \
    connect, point_from_pose, get_link_pose, link_from_name, add_line, \
    wait_for_duration, disconnect, elapsed_time, reset_simulation, wait_for_user, set_camera_pose

from extrusion.utils import get_disabled_collisions, MotionTrajectory, load_world, PrintTrajectory, is_ground, \
    TOOL_NAME
from extrusion.visualization import draw_ordered, set_extrusion_camera
from extrusion.stream import SELF_COLLISIONS

JOINT_WEIGHTS = [0.3078557810844393, 0.443600199302506, 0.23544367607317915,
                 0.03637161028426032, 0.04644626184081511, 0.015054267683041092]

def compute_motion(robot, fixed_obstacles, element_bodies, printed_elements, start_conf, end_conf, collisions=True):
    weights = np.array(JOINT_WEIGHTS)
    resolutions = np.divide(0.005*np.ones(weights.shape), weights)
    disabled_collisions = get_disabled_collisions(robot)
    movable_joints = get_movable_joints(robot)
    set_joint_positions(robot, movable_joints, start_conf)
    obstacles = fixed_obstacles + [element_bodies[e] for e in printed_elements]
    if not collisions:
        obstacles = []
    path = plan_joint_motion(robot, movable_joints, end_conf, obstacles=obstacles,
                             self_collisions=SELF_COLLISIONS, disabled_collisions=disabled_collisions,
                             weights=weights, resolutions=resolutions,
                             restarts=50, iterations=100, smooth=100)
    if path is None:
        print('Failed to find a motion plan!')
        return None
    motion_traj = MotionTrajectory(robot, movable_joints, path)
    return motion_traj

def compute_motions(robot, fixed_obstacles, element_bodies, initial_conf, print_trajectories, **kwargs):
    # TODO: can also just plan to initial conf and then shortcut
    # TODO: backoff motion
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
        motion_traj = compute_motion(robot, fixed_obstacles, element_bodies,
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
                current_point = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_NAME)))
                if last_point is not None:
                    color = (0, 0, 1) if is_ground(trajectory.element, ground_nodes) else (1, 0, 0)
                    handles.append(add_line(last_point, current_point, color=color))
                last_point = current_point
            wait_for_duration(time_step)
        #wait_for_user()

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
