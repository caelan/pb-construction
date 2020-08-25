from __future__ import print_function

import time
import numpy as np
import colorsys
import random

from scipy.spatial.qhull import QhullError

from pybullet_tools.utils import get_movable_joints, elapsed_time, wait_for_user, convex_hull, \
    create_mesh, apply_alpha, remove_body, pairwise_collision, get_sample_fn, get_distance_fn, get_extend_fn, \
    check_initial_end, birrt, INF, BASE_LINK, vertices_from_link, apply_affine, get_pose, has_gui, set_color, remove_all_debug, \
    RED

from extrusion.utils import get_disabled_collisions, MotionTrajectory, PrintTrajectory, RESOLUTION, JOINT_WEIGHTS
from extrusion.stream import get_element_collision_fn

MIN_ELEMENTS = INF # 2 | 3 | INF
LAZY = True

def create_bounding_mesh(element_bodies, node_points, printed_elements):
    # TODO: use bounding boxes instead of points
    # TODO: connected components
    # TODO: buffer or use distance from the mesh
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

def compute_motion(robot, fixed_obstacles, element_bodies,
                   printed_elements, start_conf, end_conf,
                   collisions=True, max_time=INF, smooth=100):
    # TODO: can also just plan to initial conf and then shortcut
    joints = get_movable_joints(robot)
    assert len(joints) == len(end_conf)
    weights = JOINT_WEIGHTS
    resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
    disabled_collisions = get_disabled_collisions(robot)
    custom_limits = {}
    #element_from_body = {b: e for e, b in element_bodies.items()}

    element_obstacles = {element_bodies[e] for e in printed_elements}
    obstacles = set(fixed_obstacles) | element_obstacles
    hulls = {}

    # # TODO: precompute this
    # resolution = 0.25
    # frequencies = {}
    # for element in printed_elements:
    #     #key = None
    #     midpoint = np.average([node_points[n] for n in element], axis=0)
    #     #key = int(midpoint[2] / resolution)
    #     key = tuple((midpoint / resolution).astype(int).tolist()) # round or int?
    #     frequencies.setdefault(key, []).append(element)
    # #print(len(frequencies))
    # #print(Counter({key: len(elements) for key, elements in frequencies.items()}))
    #
    # # TODO: apply this elsewhere
    # obstacles = list(fixed_obstacles)
    # for elements in frequencies.values():
    #     element_obstacles = randomize(element_bodies[e] for e in elements)
    #     if MIN_ELEMENTS <= len(elements):
    #         hull = create_bounding_mesh(element_bodies, node_points, elements)
    #         assert hull is not None
    #         hulls[hull] = element_obstacles
    #     else:
    #         obstacles.extend(element_obstacles)

    if not collisions:
        hulls = {}
        obstacles = []
    #print(hulls)
    #print(obstacles)
    #wait_for_user()

    sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(robot, joints, weights=weights)
    extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    #collision_fn = get_collision_fn(robot, joints, obstacles, attachments={}, self_collisions=SELF_COLLISIONS,
    #                                disabled_collisions=disabled_collisions, custom_limits=custom_limits, max_distance=0.)
    collision_fn = get_element_collision_fn(robot, obstacles)

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
                     restarts=50, iterations=100, smooth=smooth, max_time=max_time)

    # path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles,
    #                          self_collisions=SELF_COLLISIONS, disabled_collisions=disabled_collisions,
    #                          weights=weights, resolutions=resolutions,
    #                          restarts=50, iterations=100, smooth=100)

    for hull in hulls:
        remove_body(hull)
    if path is None:
        print('Failed to find a motion plan!')
        return None
    return MotionTrajectory(robot, joints, path)

def compute_motions(robot, fixed_obstacles, element_bodies, initial_conf, print_trajectories, **kwargs):
    # TODO: reoptimize for the sequence that have the smallest movements given this
    # TODO: sample trajectories
    # TODO: more appropriate distance based on displacement/volume
    if print_trajectories is None:
        return None
    #if any(isinstance(print_traj, MotionTrajectory) for print_traj in print_trajectories):
    #    return print_trajectories
    start_time = time.time()
    printed_elements = []
    all_trajectories = []
    current_conf = initial_conf
    for i, print_traj in enumerate(print_trajectories):
        if not np.allclose(current_conf, print_traj.start_conf, rtol=0, atol=1e-8):
            motion_traj = compute_motion(robot, fixed_obstacles, element_bodies,
                                         printed_elements, current_conf, print_traj.start_conf, **kwargs)
            if motion_traj is None:
                return None
            print('{}) {} | Time: {:.3f}'.format(i, motion_traj, elapsed_time(start_time)))
            all_trajectories.append(motion_traj)
        if isinstance(print_traj, PrintTrajectory):
            printed_elements.append(print_traj.element)
        all_trajectories.append(print_traj)
        current_conf = print_traj.end_conf

    motion_traj = compute_motion(robot, fixed_obstacles, element_bodies,
                                 printed_elements, current_conf, initial_conf, **kwargs)
    if motion_traj is None:
        return None
    return all_trajectories + [motion_traj]

##################################################

def validate_trajectories(element_bodies, fixed_obstacles, trajectories):
    if trajectories is None:
        return False
    # TODO: combine all validation procedures
    remove_all_debug()
    for body in element_bodies.values():
        set_color(body, np.zeros(4))

    print('Trajectories:', len(trajectories))
    obstacles = list(fixed_obstacles)
    for i, trajectory in enumerate(trajectories):
        for _ in trajectory.iterate():
            #wait_for_user()
            if any(pairwise_collision(trajectory.robot, body) for body in obstacles):
                if has_gui():
                    print('Collision on trajectory {}'.format(i))
                    wait_for_user()
                return False
        if isinstance(trajectory, PrintTrajectory):
            body = element_bodies[trajectory.element]
            set_color(body, apply_alpha(RED))
            obstacles.append(body)
    return True
