#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import numpy as np
import sys
sys.path.extend([
    'pddlstream/',
    'ss-pybullet/',
])

from collections import OrderedDict

from extrusion.experiment import EXCLUDE, Configuration, EXPERIMENTS_DIR
from extrusion.parsing import get_extrusion_path, load_extrusion
from extrusion.figure import bar_graph, SUCCESS, RUNTIME, SCORES, scatter_plot
from pddlstream.utils import INF, str_from_object, get_python_version
from pybullet_tools.utils import read_pickle, implies

# Geometric: python3 -m extrusion.run -l experiments/19-08-09_01-58-34.pk3
# CFree: python3 -m extrusion.run -l experiments/19-08-14_10-46-35.pk3
# Disable: python3 -m extrusion.run -l experiments/19-08-14_01-33-13.pk3

# Motions: python3 -m extrusion.analyze experiments/20-01-07_17-39-48.pk3

ALL = 'all'

ATTRIBUTES = SCORES + ['valid', 'safe', 'num_evaluated', 'min_remaining',
              'max_backtrack', 'transit_fails', 'max_translation', 'max_rotation']

##################################################

def score_result(result):
    return '{{success={:.3f}, runtime={:.0f}, valid={:.3f}, safe={:.3f}, evaluated={:.0f}, ' \
           'remaining={:.1f}, backtrack={:.1f}, transit_failures={:.1f}, max_trans={:.3E}, max_rot={:.3E}}}'.format(
            result[SUCCESS], result.get(RUNTIME, 0), result.get('valid', 0), result.get('safe', 0),
            result.get('num_evaluated', 0), result.get('min_remaining', 0), result.get('max_backtrack', 0),
            result.get('transit_fails', 0), result.get('max_translation', 0), result.get('max_rotation', 0))


def is_number(value):
    #return isinstance(value, bool) or isinstance(value, int) or isinstance(value, float)
    try:
        float(value)
        return True
    except TypeError:
        return False

##################################################

# Runtime options:
# 1) Failure is max_time
# 2) Omit failed runtimes
# 3) Average over problems all solved
# 4) Average relative for each problem solved

def load_experiment(filename, overall=False, failed_runtimes=True):
    # TODO: maybe just pass the random seed as a separate arg
    # TODO: aggregate over all problems and score using IPC rules
    # https://ipc2018-classical.bitbucket.io/
    max_time = 0
    data_from_problem = OrderedDict()
    data = read_pickle(filename)
    for config, result in data:
        #config.problem = extrusion_name_from_path(config.problem)
        if config.problem in EXCLUDE:
            continue
        problem = ALL if overall else config.problem
        plan = result.get('sequence', None)
        result[SUCCESS] = (plan is not None)
        result[SUCCESS] *= 100
        result[RUNTIME] = min(result[RUNTIME], config.max_time)
        result['length'] = len(plan) if result[SUCCESS] else INF
        #max_trans, max_rot = max_plan_deformation(config.problem, plan)
        #result['max_trans'] = max_trans
        #result['max_rot'] = max_rot
        result.pop('sequence', None)
        max_time = max(max_time, result[RUNTIME])
        if not result[SUCCESS] and not failed_runtimes:
            result.pop(RUNTIME, None)
        data_from_problem.setdefault(problem, []).append((config, result))

    for p_idx, problem in enumerate(sorted(data_from_problem)):
        print()
        problem_name = os.path.basename(os.path.abspath(problem)) # TODO: this isn't a path...
        print('{}) Problem: {}'.format(p_idx, problem_name))
        if problem != ALL:
            extrusion_path = get_extrusion_path(problem)
            element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path, verbose=False)
            print('Nodes: {} | Ground: {} | Elements: {}'.format(
                len(node_points), len(ground_nodes), len(element_from_id)))

        data_from_config = OrderedDict()
        value_per_field = {}
        for config, result in data_from_problem[problem]:
            new_config = Configuration(None, None, *config[2:])
            #print(config._asdict()) # config.__dict__
            for field, value in config._asdict().items():
                value_per_field.setdefault(field, set()).add(value)
            data_from_config.setdefault(new_config, []).append(result)

        print('Attributes:', str_from_object(value_per_field))
        print('Configs:', len(data_from_config))

        all_results = {}
        for c_idx, config in enumerate(sorted(data_from_config, key=str)):
            results = data_from_config[config]
            accumulated_result = {}
            for result in results:
                for name, value in result.items():
                    #if result[SUCCESS] or (name == SUCCESS):
                    if is_number(value):
                        accumulated_result.setdefault(name, []).append(value)
            mean_result = {name: round(np.average(values), 3) for name, values in accumulated_result.items()}
            key = {field: value for field, value in config._asdict().items()
                   if (value is not None) and (2 <= len(value_per_field[field]))}
            all_results[frozenset(key.items())] = {name: values for name, values in accumulated_result.items() if name in SCORES}
            score = score_result(mean_result)
            print('{}) {} ({}): {}'.format(c_idx, str_from_object(key), len(results), str_from_object(score)))

        if problem == ALL:
            for attribute in SCORES:
                bar_graph(all_results, attribute)
    scatter_plot(data)
    print('Max time: {:.3f} sec'.format(max_time))

##################################################

def enumerate_experiments():
    for filename in sorted(os.listdir(EXPERIMENTS_DIR)):
        path = os.path.join(EXPERIMENTS_DIR, filename)
        try:
            data = read_pickle(path)
            configs, results = zip(*data)
            #print(results[0]['sequence'])
            #problems = {config.problem for config in configs}
            algorithms = {config.algorithm for config in configs}
            heuristics = {config.bias for config in configs}
            print()
            print(path, len(data), sorted(algorithms), sorted(heuristics))
            print(configs[0])
        except TypeError:
            print('Unable to load', path)
            continue

##################################################

def main():
    assert get_python_version() == 3
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Analyze an experiment')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Enables the viewer during planning')
    parser.add_argument('-d', '--dump', action='store_true',
                        help='Dumps the configuration for each ')
    args = parser.parse_args()
    np.set_printoptions(precision=3)
    if args.dump:
        enumerate_experiments()
    load_experiment(args.path, overall=args.all)

if __name__ == '__main__':
    main()
