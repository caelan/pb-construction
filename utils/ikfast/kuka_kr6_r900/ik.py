from examples.pybullet.utils.pybullet_tools.ikfast.utils import compute_forward_kinematics, compute_inverse_kinematics, select_solution
from examples.pybullet.utils.pybullet_tools.utils import multiply, get_link_pose, get_movable_joints, \
    link_from_name, get_joint_positions, invert, violates_limits, get_pose, get_distance

BASE_FRAME = 'robot_base_link'
IK_FRAME = 'robot_tool0'
TOOL_FRAME = 'eef_tcp_frame' #robot_tool0 | eef_tcp_frame

#####################################

def get_tool_pose(robot):
    from .ikfast_kuka_kr6_r900 import get_fk
    ik_joints = get_movable_joints(robot)
    conf = get_joint_positions(robot, ik_joints)
    # TODO: this should be linked to ikfast's get numOfJoint function
    assert len(conf) == 6
    base_from_tool = compute_forward_kinematics(get_fk, conf)
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_FRAME))
    return multiply(world_from_base, base_from_tool)

#####################################

def get_tool_from_ik(robot):
    world_from_tool = get_link_pose(robot, link_from_name(robot, TOOL_FRAME))
    world_from_ik = get_link_pose(robot, link_from_name(robot, IK_FRAME))
    # tool from the bare flange (6th axis)
    return multiply(invert(world_from_tool), world_from_ik)

def get_ik_generator(robot, tool_pose):
    from .ikfast_kuka_kr6_r900 import get_ik
    world_from_base = get_link_pose(robot, link_from_name(robot, BASE_FRAME))
    #world_from_base == get_pose(robot)
    base_from_tool = multiply(invert(world_from_base), tool_pose)
    base_from_ik = multiply(base_from_tool, get_tool_from_ik(robot))
    yield compute_inverse_kinematics(get_ik, base_from_ik)

def sample_tool_ik(robot, tool_pose, closest_only=False, **kwargs):
    generator = get_ik_generator(robot, tool_pose)
    ik_joints = get_movable_joints(robot)
    solutions = next(generator)
    if closest_only and solutions:
        current_conf = get_joint_positions(robot, ik_joints)
        solutions = [min(solutions, key=lambda conf: get_distance(current_conf, conf))]
    solutions = list(filter(lambda conf: not violates_limits(robot, ik_joints, conf), solutions))
    return select_solution(robot, ik_joints, solutions, **kwargs)
