from examples.pybullet.utils.pybullet_tools.utils import joints_from_names, joint_from_name

ARM_NAMES = ['left', 'right']

# dict name composers
def arm_joints_from_arm(arm):
    assert (arm in ARM_NAMES)
    return '{}_arm'.format(arm)

def torso_from_arm(arm):
    assert (arm in ARM_NAMES)
    return '{}_torso'.format(arm)

def prefix_from_arm(arm):
    return arm[0]

ETH_RFL_GROUPS = {
        'base': ['gantry_x_joint'],
        torso_from_arm('left'): ['l_gantry_z_joint'], # 'l_gantry_y_joint'
        torso_from_arm('right'): ['r_gantry_z_joint'], # 'r_gantry_y_joint'
        arm_joints_from_arm('left'): ['l_robot_joint_{}'.format(i+1) for i in range(6)],
        arm_joints_from_arm('right'): ['r_robot_joint_{}'.format(i+1) for i in range(6)],
}

ETH_RFL_TOOL_FRAMES = {
    'left': 'l_eef_tcp_frame',
    'right': 'r_eef_tcp_frame',
}

def get_torso_arm_joint_names(arm):
    # frame name (urdf)
    return ETH_RFL_GROUPS[torso_from_arm(arm)] + \
           ETH_RFL_GROUPS[arm_joints_from_arm(arm)]

def get_arm_joint_names(arm):
    return ['{}_gantry_y_joint'.format(prefix_from_arm(arm))] + \
           get_torso_arm_joint_names(arm)

def get_torso_arm_joints(robot, arm):
    # joint id in pybullet
    return joints_from_names(robot, get_torso_arm_joint_names(arm))

def get_torso_joint(robot, arm):
    [torso_joint] = ETH_RFL_GROUPS[torso_from_arm(arm)]
    return joint_from_name(robot, torso_joint)

def get_tool_frame(arm):
    return ETH_RFL_TOOL_FRAMES[arm]
