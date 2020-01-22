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

from collections import OrderedDict, Counter

from extrusion.experiment import EXCLUDE, Configuration, EXPERIMENTS_DIR
from extrusion.parsing import get_extrusion_path, load_extrusion
from pddlstream.utils import INF, str_from_object, get_python_version
from pybullet_tools.utils import read_pickle

import pandas as pd

# Geometric: python3 -m extrusion.run -l experiments/19-08-09_01-58-34.pk3
# CFree: python3 -m extrusion.run -l experiments/19-08-14_10-46-35.pk3
# Disable: python3 -m extrusion.run -l experiments/19-08-14_01-33-13.pk3

# Motions: python3 -m extrusion.analyze experiments/20-01-07_17-39-48.pk3

ALL = 'all'

##################################################

def score_result(result):
    return '{{failure={:.3f}, runtime={:.0f}, valid={:.3f}, safe={:.3f}, evaluated={:.0f}, ' \
           'remaining={:.1f}, backtrack={:.1f}, transit_failures={:.1f}, max_trans={:.3E}, max_rot={:.3E}}}'.format(
            (1. - result['success']), result.get('runtime', 0), result.get('valid', 0), result.get('safe', 0),
            result.get('num_evaluated', 0), result.get('min_remaining', 0), result.get('max_backtrack', 0),
            result.get('transit_fails', 0), result.get('max_translation', 0), result.get('max_rotation', 0))

def load_experiment(filename, overall=False, write_report=False):
    # TODO: maybe just pass the random seed as a separate arg
    # TODO: aggregate over all problems and score using IPC rules
    # https://ipc2018-classical.bitbucket.io/
    max_time = 0
    data_from_problem = OrderedDict()
    for config, result in read_pickle(filename):
        #config.problem = extrusion_name_from_path(config.problem)
        if config.problem in EXCLUDE:
            continue
        problem = ALL if overall else config.problem
        plan = result.get('sequence', None)
        result['success'] = (plan is not None)
        result['length'] = len(plan) if result['success'] else INF
        #max_trans, max_rot = max_plan_deformation(config.problem, plan)
        #result['max_trans'] = max_trans
        #result['max_rot'] = max_rot
        result.pop('sequence', None)
        if result['success']:
            max_time = max(max_time, result['runtime'])
        data_from_problem.setdefault(problem, []).append((config, result))

    column_names = ('config_id','shape','info','algorithm','bias',
                    'success','runtime','num_evaluated','min_remaining','max_backtrack','max_translation','max_rotation', 'length', 'num_elements', 'transit_failures', 'stiffness_failures', 'valid', 'safe', 'seed')
    df = pd.DataFrame(columns=column_names)
    if not overall:
        shown_column_names = ('algorithm','bias',
                              'success','runtime','num_evaluated','min_remaining','max_backtrack','max_translation','max_rotation', 'length', 'num_elements', 'safe', 'transit_failures', 'stiffness_failures', 'valid')
        col_name_df = {cn : cn for cn in shown_column_names}    

    for p_idx, problem in enumerate(sorted(data_from_problem)):
        print()
        problem_name = os.path.basename(os.path.abspath(problem)) # TODO: this isn't a path...
        print('{}) Problem: {}'.format(p_idx, problem_name))

        if problem != ALL:
            extrusion_path = get_extrusion_path(problem)
            element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path, verbose=False)
            print('Nodes: {} | Ground: {} | Elements: {}'.format(
                len(node_points), len(ground_nodes), len(element_from_id)))

            df = df.append(pd.Series(), ignore_index=True)
            df = df.append([{ 'config_id' : p_idx,
                              'shape' : problem_name, 
                              'info' : 'Nodes: {} | Ground: {} | Elements: {}'.format(
                                       len(node_points), len(ground_nodes), len(element_from_id))
                 }])

        data_from_config = OrderedDict()
        value_per_field = {}
        for config, result in data_from_problem[problem]:
            new_config = Configuration(None, None, *config[2:])
            #print(config._asdict()) # config.__dict__
            for field, value in config._asdict().items():
                value_per_field.setdefault(field, set()).add(value)
            data_from_config.setdefault(new_config, []).append(result)

        # * config print at each problem
        print('Attributes:', str_from_object(value_per_field))
        df = df.append([{ 'info' : str_from_object(
            {field : value_per_field[field] for field in value_per_field.keys() if field not in \
                ['algorithm', 'bias', 'seed', 'problem']})
            }])
        if not overall:
            df = df.append([col_name_df])

        print('Configs:', len(data_from_config))
        for c_idx, config in enumerate(sorted(data_from_config, key=str)):
            results = data_from_config[config]
            accumulated_result = {}
            for result in results:
                for name, value in result.items():
                    #if result['success'] or (name == 'success'):
                    if isinstance(value, int) or isinstance(value, float):
                        accumulated_result.setdefault(name, []).append(value)
            mean_result = {name: round(np.average(values), 3) for name, values in accumulated_result.items()}
            key = {field: value for field, value in config._asdict().items()
                   if 2 <= len(value_per_field[field])}
            score = score_result(mean_result)
            print('{}) {} ({}): {}'.format(c_idx, str_from_object(key), len(results), str_from_object(score)))
            print('Max time: {:.3f} sec'.format(max_time))

            df_data = {}
            df_data.update({'config_id' : c_idx})
            df_data.update(key)
            df_data.update(mean_result)
            # print(df_data)
            df = df.append(df_data, ignore_index=True)
    return df
##################################################

def enumerate_experiments():
    for filename in sorted(os.listdir(EXPERIMENTS_DIR)):
        path = os.path.join(EXPERIMENTS_DIR, filename)
        try:
            data = read_pickle(path)
            configs, _ = zip(*data)
            #problems = {config.problem for config in configs}
            algorithms = {config.algorithm for config in configs}
            print(path, sorted(algorithms))
        except TypeError:
            print('Unable to load', path)
            continue

##################################################

def main():
    assert get_python_version() == 3
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='The absolute path to an experiment pickle file')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Print out summary report for all problems')
    parser.add_argument('-w', '--write_report', action='store_true',
                        help='Write a spreadsheet report for the experiment.')
    args = parser.parse_args()
    np.set_printoptions(precision=3)
    #enumerate_experiments()
    df = load_experiment(args.path, overall=args.all)

    if args.write_report:
        exp_dir = os.path.dirname(args.path)
        csv_file_path = os.path.join(exp_dir, args.path.split(os.path.sep)[-1].split('.')[0] + '.xlsx')
        df2 = load_experiment(args.path, overall=not args.all)

        # df.to_csv(csv_file_path)
        with pd.ExcelWriter(csv_file_path) as writer:
            df.to_excel(writer, sheet_name='detailed' if not args.all else 'summary')
            df2.to_excel(writer, sheet_name='summary' if not args.all else 'detailed')

if __name__ == '__main__':
    main()
