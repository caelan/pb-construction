import os
import numpy as np

from collections import namedtuple, OrderedDict

from extrusion.parsing import load_extrusion, get_extrusion_path, extrusion_name_from_path
from extrusion.utils import evaluate_stiffness, create_stiffness_checker, TRANS_TOL, ROT_TOL
from pddlstream.utils import str_from_object, INF
from examples.pybullet.utils.pybullet_tools.utils import read_pickle

Configuration = namedtuple('Configuration', ['seed', 'problem', 'algorithm', 'bias', 'max_time',
                                             'cfree', 'disable', 'stiffness', 'motions'])
#Score = namedtuple('Score', ['failure', 'runtime', 'max_trans', 'max_rot'])


EXCLUDE = [
    'rotated_dented_cube',
    'robarch_tree',
    'DJMM_bridge',
]

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

def load_experiment(filename, overall=False):
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