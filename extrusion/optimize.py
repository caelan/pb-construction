from __future__ import print_function

import random
import time
from collections import namedtuple

from extrusion.parsing import load_extrusion
from extrusion.stream import get_print_gen_fn, MAX_DIRECTIONS, MAX_ATTEMPTS
from extrusion.utils import get_cspace_distance
from pybullet_tools.utils import INF
from pybullet_tools.utils import elapsed_time

OPTIMIZE = False

#State = namedtuple('State', ['element', 'printed', 'plan'])
Node = namedtuple('Node', ['action', 'state'])

# TODO: sample configuration that connects adjacent ones

def get_command_distance(robot, initial_conf, commands, index, new_command):
    assert 0 <= index <= len(commands) - 1
    last_conf = initial_conf if index == 0 else commands[index - 1].end_conf
    next_conf = initial_conf if index == len(commands) - 1 else commands[index + 1].start_conf
    return get_cspace_distance(robot, last_conf, new_command.start_conf) + \
           new_command.get_distance() + \
           get_cspace_distance(robot, new_command.end_conf, next_conf)

def optimize_commands(robot, obstacles, element_bodies, extrusion_path, initial_conf, commands,
                      max_iterations=INF, max_time=60,
                      motions=True, collisions=True, **kwargs):
    if commands is None:
        return None
    start_time = time.time()
    element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
                                    supports=False, precompute_collisions=False,
                                    max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS,
                                    collisions=collisions, **kwargs)

    commands = list(commands)
    sequence = [command.elements[0] for command in commands]
    directed_sequence = [command.directed_elements[0] for command in commands]
    indices = list(range(len(sequence)))

    #trajectories = flatten_commands(commands)
    #print(len(trajectories), get_print_distance(trajectories, teleport=False))

    total_distance = sum(get_command_distance(robot, initial_conf, commands, index, command)
                         for index, command in enumerate(commands))
    # TODO: bias sampling towards far apart relative to neighoors
    # TODO: bias IK solutions to be the best of several

    iterations = extrusion_failures = 0
    while (iterations < max_iterations) and (elapsed_time(start_time) < max_time):
        iterations += 1
        index = random.choice(indices)
        printed = sequence[:index-1]
        element = sequence[index]
        directed = directed_sequence[index]

        print(commands[index])
        distance = get_command_distance(robot, initial_conf, commands, index, commands[index])
        print('Iteration {} | Failures: {} | Distance: {:.3f} | Index: {}/{} | Time: {:.3f}'.format(
            iterations, extrusion_failures, total_distance, index, len(indices), elapsed_time(start_time)))

        new_command, = next(print_gen_fn(directed[0], element, extruded=printed), (None,))
        if new_command is None:
            extrusion_failures += 1
            continue
        new_distance = get_command_distance(robot, initial_conf, commands, index, new_command)
        if new_distance < distance:
            commands.pop(index)
            commands.insert(index, new_command)
            total_distance -= (distance - new_distance)

    # data = {
    #     'extrusion_failures': extrusion_failures,
    # }
    return commands #, data
