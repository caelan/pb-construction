#!/usr/bin/env python
from __future__ import print_function

import sys
import argparse
import cProfile
import pstats
import numpy as np
import time
import os
from termcolor import cprint

sys.path.extend([
    'pddlstream/',
    'pyplanners/',
    'ss-pybullet/',
    'extrusion/',
])

import pddlstream
cprint('Using PDDLStream from {}'.format(pddlstream.__file__), 'yellow')
import strips # pyplanners
cprint('Using strips from {}'.format(strips.__file__), 'yellow')

from pybullet_tools.utils import connect, disconnect, get_movable_joints, get_joint_positions, LockRenderer, \
    unit_pose, reset_simulation, draw_pose, apply_alpha, BLACK, Pose, Euler, has_gui, wait_for_user, wait_if_gui

##################################################

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='simple_frame',
                        help='The name of the problem to solve')
    parser.add_argument('-v', '--viewer', action='store_true',
                        help='Enables the viewer during planning')
    args = parser.parse_args()

    wait_for_user()

if __name__ == '__main__':
    main()