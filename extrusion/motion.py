from __future__ import print_function

import time
import numpy as np
import colorsys
import random

from scipy.spatial.qhull import QhullError

from pybullet_tools.utils import get_movable_joints, elapsed_time, wait_for_user, convex_hull, \
    create_mesh, apply_alpha, remove_body, pairwise_collision, get_sample_fn, get_distance_fn, get_extend_fn, \
    check_initial_end, birrt, INF, BASE_LINK, vertices_from_link, apply_affine, get_pose, has_gui, set_color, remove_all_debug, \
    RED, randomize, get_refine_fn, AABB, get_aabb_vertices, set_joint_positions

from extrusion.utils import get_disabled_collisions, MotionTrajectory, PrintTrajectory, RESOLUTION, JOINT_WEIGHTS
from extrusion.stream import get_element_collision_fn

MIN_ELEMENTS = INF # 2 | 3 | INF
LAZY = True

def get_pairs(iterator):
    try:
        last = next(iterator)
    except StopIteration:
        return
    for current in iterator:
        yield last, current
        last = current

def create_bounding_mesh(printed_elements, element_bodies=None, node_points=None, buffer=0.):
    # TODO: use bounding boxes instead of points
    # TODO: connected components
    # TODO: implicit buffer by distance from the mesh
    assert printed_elements
    assert element_bodies or node_points
    printed_points = []
    if node_points is not None:
        printed_points.extend(node_points[n] for element in printed_elements for n in element)
    if element_bodies is not None:
        for element in printed_elements:
            body = element_bodies[element]
            printed_points.extend(apply_affine(get_pose(body), vertices_from_link(body, BASE_LINK)))

    if buffer != 0.:
        half_extents = buffer*np.ones(3) / 2.
        for point in list(printed_points):
            printed_points.extend(np.array(point) + np.array(corner)
                                  for corner in get_aabb_vertices(AABB(-half_extents, half_extents)))

    rgb = colorsys.hsv_to_rgb(h=random.random(), s=1, v=1)
    #rgb = RED
    try:
        mesh = convex_hull(printed_points)
        # handles = draw_mesh(mesh)
        return create_mesh(mesh, under=True, color=apply_alpha(rgb, 0.5))
    except QhullError as e:
        print(printed_elements)
        raise e
        #return None

def decompose_structure(fixed_obstacles, element_bodies, printed_elements,  resolution=0.25):
    # TODO: precompute this
    frequencies = {}
    for element in printed_elements:
        #key = None
        midpoint = np.average([node_points[n] for n in element], axis=0)
        #key = int(midpoint[2] / resolution)
        key = tuple((midpoint / resolution).astype(int).tolist()) # round or int?
        frequencies.setdefault(key, []).append(element)
    #print(len(frequencies))
    #print(Counter({key: len(elements) for key, elements in frequencies.items()}))

    # TODO: apply this elsewhere
    hulls = {}
    obstacles = list(fixed_obstacles)
    for elements in frequencies.values():
        element_obstacles = randomize(element_bodies[e] for e in elements)
        if MIN_ELEMENTS <= len(elements):
            hull = create_bounding_mesh(element_bodies, node_points, elements)
            assert hull is not None
            hulls[hull] = element_obstacles
        else:
            obstacles.extend(element_obstacles)

    #pairwise_collision(robot, element, max_distance=0.1) # body_collision
    return hulls, obstacles

##################################################

def compute_motion(robot, fixed_obstacles, element_bodies,
                   printed_elements, start_conf, end_conf,
                   collisions=True, max_time=INF, buffer=0.1, smooth=100):
    # TODO: can also just plan to initial conf and then shortcut
    joints = get_movable_joints(robot)
    assert len(joints) == len(end_conf)
    weights = JOINT_WEIGHTS
    resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
    disabled_collisions = get_disabled_collisions(robot)
    custom_limits = {}
    #element_from_body = {b: e for e, b in element_bodies.items()}

    hulls, obstacles = {}, []
    if collisions:
        element_obstacles = {element_bodies[e] for e in printed_elements}
        obstacles = set(fixed_obstacles) | element_obstacles
        #hulls, obstacles = decompose_structure(fixed_obstacles, element_bodies, printed_elements)
    #print(hulls)
    #print(obstacles)
    #wait_for_user()

    #printed_elements = set(element_bodies)
    bounding = None
    if printed_elements:
        # TODO: pass in node_points
        bounding = create_bounding_mesh(printed_elements, element_bodies=element_bodies, node_points=None, buffer=buffer)
        #wait_for_user()

    sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(robot, joints, weights=weights)
    extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    #collision_fn = get_collision_fn(robot, joints, obstacles, attachments={}, self_collisions=SELF_COLLISIONS,
    #                                disabled_collisions=disabled_collisions, custom_limits=custom_limits, max_distance=0.)
    collision_fn = get_element_collision_fn(robot, obstacles)

    fine_extend_fn = get_extend_fn(robot, joints, resolutions=1e-1*resolutions) #, norm=INF)

    def test_bounding(q):
        set_joint_positions(robot, joints, q)
        collision = (bounding is not None) and pairwise_collision(robot, bounding)
        return q, collision

    def dynamic_extend_fn(q_start, q_end):
        # TODO: retime trajectories to be move more slowly around the structure
        for (q1, c1), (q2, c2) in get_pairs(map(test_bounding, extend_fn(q_start, q_end))):
            # print(c1, c2, len(list(fine_extend_fn(q1, q2))))
            # set_joint_positions(robot, joints, q2)
            # wait_for_user()
            if c1 and c2:
                for q in fine_extend_fn(q1, q2):
                    # set_joint_positions(robot, joints, q)
                    # wait_for_user()
                    yield q
            else:
                yield q2

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
        path = birrt(start_conf, end_conf, distance_fn, sample_fn, dynamic_extend_fn, element_collision_fn,
                     restarts=50, iterations=100, smooth=smooth, max_time=max_time)
    # path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles,
    #                          self_collisions=SELF_COLLISIONS, disabled_collisions=disabled_collisions,
    #                          weights=weights, resolutions=resolutions,
    #                          restarts=50, iterations=100, smooth=100)

    # if (bounding is not None) and (path is not None):
    #     for i, q in enumerate(path):
    #         print('{}/{}'.format(i, len(path)))
    #         set_joint_positions(robot, joints, q)
    #         wait_for_user()

    if bounding is not None:
        remove_body(bounding)
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
