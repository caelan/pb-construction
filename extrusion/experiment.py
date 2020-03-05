import datetime
import os
import time
import sys

sys.path.extend([
    'pddlstream/',
    'ss-pybullet/',
])

from itertools import product
from multiprocessing import cpu_count, Pool
#from multiprocessing.context import TimeoutError

from collections import namedtuple

from extrusion.progression import progression
from extrusion.lookahead import lookahead
from extrusion.heuristics import DISTANCE_HEURISTICS, COST_HEURISTICS, HEURISTICS
from extrusion.parsing import enumerate_problems
from extrusion.regression import regression
from extrusion.stream import SKIP_PERCENTAGE
from pddlstream.utils import get_python_version
from pybullet_tools.utils import is_darwin, user_input, write_pickle, elapsed_time, DATE_FORMAT, chunks

# TODO: use dicts instead

Configuration = namedtuple('Configuration', ['seed', 'problem', 'algorithm', 'bias', 'max_time',
                                             'cfree', 'disable', 'stiffness', 'motions', 'ee_only'])
#Score = namedtuple('Score', ['failure', 'runtime', 'max_trans', 'max_rot'])

LOOKAHEAD_ALGORITHMS = [lookahead.__name__]

ALGORITHMS = [alg.__name__ for alg in [progression, lookahead, regression]] #+ [STRIPSTREAM_ALGORITHM]

EXCLUDE = [
    #'dented_cube', # TODO: 3D_truss isn't supported error
    'rotated_dented_cube', # Structure violates stiffness
    'robarch_tree', # Cannot print ground elements
    'DJMM_bridge', # Too large for pybullet
    'klein_bottle_trail', # Structure violates stiffness
]

EXPERIMENTS_DIR = 'experiments/'

##################################################

def train_parallel(args, n=1):
    from extrusion.run import plan_extrusion
    assert SKIP_PERCENTAGE == 0
    initial_time = time.time()

    problems = sorted(set(enumerate_problems()) - set(EXCLUDE))
    #problems = ['four-frame']
    #problems = ['simple_frame', 'topopt-101_tiny', 'topopt-100_S1_03-14-2019_w_layer']

    algorithms = list(ALGORITHMS)
    if args.disable:
        for algorithm in LOOKAHEAD_ALGORITHMS:
            if algorithm in algorithms:
                algorithms.remove(algorithm)
    #algorithms = ['regression']

    heuristics = HEURISTICS
    #heuristics = DISTANCE_HEURISTICS + COST_HEURISTICS

    seeds = list(range(args.num))
    if n is None:
        n = len(seeds)
    groups = list(chunks(seeds, n=n))

    print('Chunks: {}'.format(len(groups)))
    print('Problems ({}): {}'.format(len(problems), problems))
    #problems = [path for path in problems if 'simple_frame' in path]
    print('Algorithms ({}): {}'.format(len(algorithms), algorithms))
    print('Heuristics ({}): {}'.format(len(heuristics), heuristics))
    jobs = [[Configuration(seed, problem, algorithm, heuristic, args.max_time, args.cfree,
                           args.disable, args.stiffness, args.motions, args.ee_only)
                       for seed, algorithm, heuristic in product(group, algorithms, heuristics)]
                      for problem, group in product(problems, groups)]
    # TODO: separate out the algorithms again
    # TODO: print the size per job
    print('Jobs: {}'.format(len(jobs)))

    serial = is_darwin()
    available_cores = cpu_count()
    num_cores = max(1, min(1 if serial else available_cores - 4, len(jobs)))
    print('Max Cores:', available_cores)
    print('Serial:', serial)
    print('Using Cores:', num_cores)
    date = datetime.datetime.now().strftime(DATE_FORMAT)
    filename = '{}.pk{}'.format(date, get_python_version())
    path = os.path.join(EXPERIMENTS_DIR, filename)
    print('Data path:', path)

    user_input('Begin?')
    start_time = time.time()
    timeouts = 0
    pool = Pool(processes=num_cores)  # , initializer=mute)
    generator = pool.imap_unordered(plan_extrusion, jobs, chunksize=1)
    results = []
    while True:
        # TODO: randomly sort instead
        last_time = time.time()
        try:
            for config, data in generator.next(): # timeout=2 * args.max_time)
                results.append((config, data))
                print('{}/{} completed | {:.3f} seconds | timeouts: {} | {}'.format(
                    len(results), len(jobs), elapsed_time(start_time), timeouts,
                    datetime.datetime.now().strftime(DATE_FORMAT)))
                print(config, data)
            if results:
                write_pickle(path, results)
                print('Saved', path)
        except StopIteration:
            break
        # except TimeoutError:
        #     # TODO: record this as a failure? Nothing is saved though...
        #     timeouts += 1
        #     #traceback.print_exc()
        #     print('Error! Timed out after {:.3f} seconds'.format(elapsed_time(last_time)))
        #     break # This kills all jobs
        #     #continue # This repeats jobs until success
    print('Total time:', elapsed_time(initial_time))
    return results

def main():
    from extrusion.run import create_parser
    parser = create_parser()
    parser.add_argument('-n', '--num', default=3, type=int,
                        help='Number of experiment trials')
    args = parser.parse_args()
    args.viewer = False
    if args.disable:
        args.cfree = True
        args.motions = False
        args.max_time = 5*60 # 259 for duck
    print('Arguments:', args)
    train_parallel(args)

if __name__ == '__main__':
    main()

# python -m extrusion.run -n 10 2>&1 | tee log.txt
