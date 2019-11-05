import datetime
import os
import time
from itertools import product
from multiprocessing import cpu_count, Pool
from multiprocessing.context import TimeoutError

import numpy as np

from collections import namedtuple, OrderedDict

from extrusion.heuristics import HEURISTICS
from extrusion.parsing import load_extrusion, get_extrusion_path, extrusion_name_from_path, enumerate_problems
from extrusion.utils import evaluate_stiffness, create_stiffness_checker, TRANS_TOL, ROT_TOL
from pddlstream.utils import str_from_object, INF, get_python_version
from pybullet_tools.utils import read_pickle, is_darwin, user_input, write_pickle, elapsed_time

Configuration = namedtuple('Configuration', ['seed', 'problem', 'algorithm', 'bias', 'max_time',
                                             'cfree', 'disable', 'stiffness', 'motions'])
#Score = namedtuple('Score', ['failure', 'runtime', 'max_trans', 'max_rot'])


EXCLUDE = [
    'rotated_dented_cube',
    'robarch_tree',
    'DJMM_bridge',
]

# Geometric: python3 -m extrusion.run -l experiments/19-08-09_01-58-34.pk3
# CFree: python3 -m extrusion.run -l experiments/19-08-14_10-46-35.pk3
# Disable: python3 -m extrusion.run -l experiments/19-08-14_01-33-13.pk3

##################################################

def score_result(result):
    return '{{failure={:.3f}, runtime={:.0f}, evaluated={:.0f}, max_trans={:.3E}, max_rot={:.3E}}}'.format(
        1. - result['success'], result.get('runtime', 0), result.get('num_evaluated', 0),
        result.get('max_trans', 0), result.get('max_rot', 0))

def max_plan_deformation(config, result):
    plan = result.get('sequence', None)
    if plan is None:
        #return 0, 0
        return TRANS_TOL, ROT_TOL
    # TODO: absence of entry means ignore
    # TODO: inspect the number fo states searched rather than time overhead
    problem = extrusion_name_from_path(config.problem)
    problem_path = get_extrusion_path(problem)
    element_from_id, _, _ = load_extrusion(problem_path)
    checker = create_stiffness_checker(problem_path, verbose=False)
    #trans_tol, rot_tol = checker.get_nodal_deformation_tol()

    printed = []
    translations = []
    rotations = []
    for element in plan:
        printed.append(element)
        deformation = evaluate_stiffness(problem, element_from_id, printed,
                                         checker=checker, verbose=False)
        trans, rot, _, _ = checker.get_max_nodal_deformation()
        translations.append(trans)
        rotations.append(rot)
    return max(translations), max(rotations)

# Failed instances
# fertility, duck, dented_cube, compas_fea_beam_tree_M, compas_fea_beam_tree, bunny_full_tri_dense, bunny_full_quad, C_shape

ALL = 'all'

def load_experiment(filename, overall=True):
    # TODO: maybe just pass the random seed as a separate arg
    # TODO: aggregate over all problems and score using IPC rules
    # https://ipc2018-classical.bitbucket.io/
    data_from_problem = OrderedDict()
    for config, result in read_pickle(filename):
        #config.problem = extrusion_name_from_path(config.problem)
        if config.problem in EXCLUDE:
            continue
        problem = ALL if overall else config.problem
        plan = result.get('sequence', None)
        result['success'] = (plan is not None)
        result['length'] = len(plan) if result['success'] else INF
        #max_trans, max_rot = max_plan_deformation(config, result)
        #result['max_trans'] = max_trans
        #result['max_rot'] = max_rot
        result.pop('sequence', None)
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
        for c_idx, config in enumerate(sorted(data_from_config, key=str)):
            results = data_from_config[config]
            accumulated_result = {}
            for result in results:
                for name, value in result.items():
                    #if result['success'] or (name == 'success'):
                    accumulated_result.setdefault(name, []).append(value)
            mean_result = {name: round(np.average(values), 3) for name, values in accumulated_result.items()}
            key = {field: value for field, value in config._asdict().items()
                   if 2 <= len(value_per_field[field])}
            score = score_result(mean_result)
            print('{}) {} ({}): {}'.format(c_idx, str_from_object(key), len(results), str_from_object(score)))

##################################################

def train_parallel(args):
    from extrusion.run import ALGORITHMS, plan_extrusion
    initial_time = time.time()
    problems = sorted(set(enumerate_problems()) - set(EXCLUDE))
    #problems = ['simple_frame']
    print('Problems ({}): {}'.format(len(problems), problems))
    #problems = [path for path in problems if 'simple_frame' in path]
    configurations = [Configuration(*c) for c in product(
        range(args.num), problems, ALGORITHMS, HEURISTICS, [args.max_time],
        [args.cfree], [args.disable], [args.stiffness], [args.motions])]
    print('Configurations: {}'.format(len(configurations)))

    serial = is_darwin()
    available_cores = cpu_count()
    num_cores = max(1, min(1 if serial else available_cores - 3, len(configurations)))
    print('Max Cores:', available_cores)
    print('Serial:', serial)
    print('Using Cores:', num_cores)
    date = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    filename = '{}.pk{}'.format(date, get_python_version())
    path = os.path.join('experiments', filename)
    print('Data path:', path)

    user_input('Begin?')
    pool = Pool(processes=num_cores)  # , initializer=mute)
    generator = pool.imap_unordered(plan_extrusion, configurations, chunksize=1)
    results = []
    while True:
        start_time = time.time()
        try:
            configuration, data = generator.next(timeout=2 * args.max_time)
            print(len(results), configuration, data)
            results.append((configuration, data))
            if results:
                write_pickle(path, results)
                print('Saved', path)
        except StopIteration:
            break
        except TimeoutError:
            print('Error! Timed out after {:.3f} seconds'.format(elapsed_time(start_time)))
            break
    print('Total time:', elapsed_time(initial_time))
    return results