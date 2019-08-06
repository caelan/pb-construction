import os
import numpy as np

from collections import namedtuple, OrderedDict

from pddlstream.utils import str_from_object
from examples.pybullet.utils.pybullet_tools.utils import read_pickle

Configuration = namedtuple('Configuration', ['seed', 'problem', 'algorithm', 'bias', 'max_time',
                                             'cfree', 'disable', 'stiffness', 'motions'])
Score = namedtuple('Score', ['failure', 'runtime'])


def score_result(result):
    return Score(1. - round(result['success'], 3),
                 round(result.get('runtime', 0), 3))


def load_experiment(filename, overall=True):
    # TODO: maybe just pass the random seed as a separate arg
    # TODO: aggregate over all problems and score using IPC rules
    # https://ipc2018-classical.bitbucket.io/
    data_from_problem = OrderedDict()
    for config, result in read_pickle(filename):
        problem = 'all' if overall else config.problem
        data_from_problem.setdefault(problem, []).append((config, result))

    for p_idx, problem in enumerate(sorted(data_from_problem)):
        print()
        print('{}) Problem: {}'.format(p_idx, os.path.basename(os.path.abspath(problem))))

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
        for c_idx, config in enumerate(sorted(data_from_config, key=str)):
            results = data_from_config[config]
            accumulated_result = {}
            for result in results:
                for name, value in result.items():
                    accumulated_result.setdefault(name, []).append(value)
            mean_result = {name: round(np.average(values), 3) for name, values in accumulated_result.items()}
            score = score_result(mean_result)
            key = {field: value for field, value in config._asdict().items()
                                   if 2 <= len(value_per_field[field])}
            print('{}) {} ({}): {}'.format(c_idx, str_from_object(key), len(results), str_from_object(score)))