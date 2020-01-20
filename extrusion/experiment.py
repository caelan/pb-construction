import datetime
import os
import time
from itertools import product
from multiprocessing import cpu_count, Pool
from multiprocessing.context import TimeoutError

from collections import namedtuple

from extrusion.progression import progression
from extrusion.heuristics import HEURISTICS
from extrusion.parsing import enumerate_problems
from extrusion.regression import regression
from pddlstream.utils import get_python_version
from pybullet_tools.utils import is_darwin, user_input, write_pickle, elapsed_time

# TODO: use dicts instead

Configuration = namedtuple('Configuration', ['seed', 'problem', 'algorithm', 'bias', 'max_time',
                                             'cfree', 'disable', 'stiffness', 'motions', 'ee_only'])
#Score = namedtuple('Score', ['failure', 'runtime', 'max_trans', 'max_rot'])

GREEDY_ALGORITHMS = [
    progression.__name__,
    regression.__name__,
]
ALGORITHMS = GREEDY_ALGORITHMS + ['lookahead'] #+ [STRIPSTREAM_ALGORITHM]
#ALGORITHMS = ['lookahead']

EXCLUDE = [
    #'dented_cube', # TODO: 3D_truss isn't supported error
    'rotated_dented_cube', # Structure violates stiffness
    'robarch_tree', # Cannot print ground elements
    'DJMM_bridge', # Too large for pybullet
]

EXPERIMENTS_DIR = 'experiments/'

# Failed instances
# fertility, duck, dented_cube, compas_fea_beam_tree_M, compas_fea_beam_tree, bunny_full_tri_dense, bunny_full_quad, C_shape


# Can greedily print
# four-frame, simple_frame, voronoi

# Cannot greedily print
# topopt-100
# mars_bubble
# djmm_bridge
# djmm_test_block

##################################################

def train_parallel(args):
    from extrusion.run import plan_extrusion
    initial_time = time.time()
    problems = sorted(set(enumerate_problems()) - set(EXCLUDE))
    #problems = ['simple_frame']
    #algorithms = ALGORITHMS
    algorithms = ['regression']
    heuristics = HEURISTICS
    #heuristics = ['dijkstra']

    print('Problems ({}): {}'.format(len(problems), problems))
    #problems = [path for path in problems if 'simple_frame' in path]
    print('Algorithms ({}): {}'.format(len(algorithms), algorithms))
    print('Heuristics ({}): {}'.format(len(heuristics), heuristics))
    configurations = [Configuration(*c) for c in product(
        range(args.num), problems, algorithms, heuristics, [args.max_time],
        [args.cfree], [args.disable], [args.stiffness], [args.motions], [args.ee_only])]
    print('Configurations: {}'.format(len(configurations)))

    serial = is_darwin()
    available_cores = cpu_count()
    num_cores = max(1, min(1 if serial else available_cores - 3, len(configurations)))
    print('Max Cores:', available_cores)
    print('Serial:', serial)
    print('Using Cores:', num_cores)
    date = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    filename = '{}.pk{}'.format(date, get_python_version())
    path = os.path.join(EXPERIMENTS_DIR, filename)
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
