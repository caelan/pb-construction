#!/usr/bin/env python

from __future__ import print_function

import math
import scipy.stats

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
#from tabulate import tabulate

from extrusion.experiment import EXCLUDE, Configuration, EXPERIMENTS_DIR, ALGORITHMS
from extrusion.heuristics import HEURISTICS

DEFAULT_MAX_TIME = 1 * 60 * 60
#DEFAULT_MAX_TIME = 1.5 * 60 * 60

SUCCESS = 'success'
RUNTIME = 'runtime'
SCORES = [SUCCESS, RUNTIME]

FONT_SIZE = 14
WIDTH = 0.2
ALPHA = 1.0 # 0.5

##################################################

RENAME_LABELS = {
    'none': 'Random', # random
    'z': 'EuclideanDist', # task-distance
    'dijkstra': 'GraphDist', # truss-distance
    'plan-stiffness': 'StiffPlan', # stiffness-plan
    #'lookahead': 'progression+lookahead',
    SUCCESS: '% solved',
    RUNTIME: 'runtime (sec)',
    'progression': 'Progression',
    'lookahead': 'ForwardCheck',
    'regression': 'Regression',
}

def rename(name):
    return RENAME_LABELS.get(name, name)

##################################################

def bar_graph(data, attribute):
    matplotlib.rcParams.update({'font.size': FONT_SIZE})
    #pltfig, ax = plt.subplots()
    hatch = '/' if attribute == RUNTIME else None
    ax = plt.subplot()
    algorithms = sorted({dict(key)['algorithm'] for key in data} & set(ALGORITHMS), key=ALGORITHMS.index)
    print('Algorithms:', algorithms)
    heuristics = sorted({dict(key)['bias'] for key in data} & set(HEURISTICS), key=HEURISTICS.index)
    print('Heuristics:', heuristics)

    indices = np.array(range(len(algorithms)))
    for h_idx, heuristic in enumerate(heuristics): # Add everything with the same label at once
        values = []
        for algorithm in algorithms:
            key = frozenset({'algorithm': algorithm, 'bias': heuristic}.items())
            if key in data:
                values.append(data[key][attribute])
        means = list(map(np.mean, values)) # 100
        #alpha = 0.5
        #stds = list(map(np.std, values)) if attribute == RUNTIME else None
        stds = None
        rects = plt.bar(h_idx*WIDTH + indices, means, WIDTH, alpha=ALPHA, hatch=hatch, yerr=stds,
                        label=rename(heuristic)) # align='center'
    y_max = 100 if attribute == SUCCESS else DEFAULT_MAX_TIME

    #plt.title('Extrusion Planning: Stiffness Only')
    plt.title('Extrusion Planning: All Constraints')
    ticks = np.arange(len(algorithms)) + WIDTH*len(algorithms)/2.
    plt.xticks(ticks, map(rename, algorithms))
    plt.xlabel('Algorithm')
    ax.autoscale(tight=True)
    plt.legend(loc='best') # 'upper left'
    #plt.legend(loc='upper left')
    plt.ylabel(rename(attribute))
    plt.ylim([0, y_max])
    #plt.savefig('test')
    plt.tight_layout()
    #plt.grid()
    plt.show()

##################################################

#ALPHA = None
#EDGES = ['face', 'face', 'g', 'b']
#COLORS = ['r', 'y', 'none', 'none']

MARKERS = ['x', 'o', '+']
EDGES = ['face', 'g', 'face', ] # C2
#COLORS = ['C0', 'C1', 'none']
#COLORS = ['c', 'y', 'none'] # m
COLORS = ['r', 'none', 'b']


# https://matplotlib.org/api/markers_api.html
# https://matplotlib.org/2.0.2/api/colors_api.html

def scatter_plot(data):
    all_sizes = sorted({result['num_elements'] for _, result in data})
    print('Sizes:', all_sizes)
    plt.scatter(all_sizes, np.zeros(len(all_sizes)), marker='|', color='k') # black
    algorithms, heuristics = ALGORITHMS, ['z'] # plan-stiffness, z, dijkstra
    #algorithms, heuristics = ['regression'], HEURISTICS

    for a_idx, algorithm in enumerate(algorithms):
        for h_idx, heuristic in enumerate(heuristics):
            sizes = []
            runtimes = []
            for config, result in data:
                if (config.algorithm == algorithm) and (config.bias == heuristic):
                    # TODO: could hash instead
                    if result['success']:
                        sizes.append(result['num_elements'])
                        runtimes.append(result['runtime'])
            components = []
            if len(algorithms) != 1:
                components.append(rename(algorithm))
            if len(heuristics) != 1:
                components.append(rename(heuristic))
            label = '-'.join(components)
            plt.scatter(sizes, runtimes, marker=MARKERS[a_idx],
                        color=COLORS[a_idx], edgecolors=EDGES[a_idx],
                        alpha=0.75, label=label)

    plt.title('Scaling: All Constraints')
    #plt.xticks(range(1, max_size+1)) #, [get_index(problem) for problem in problems])
    plt.xlim([1, 1000]) #max(all_sizes)])
    plt.xlabel('# elements')
    plt.ylabel('runtime (sec)')
    #plt.legend(loc='upper left')
    plt.legend(loc='upper center')
    #plt.savefig('test')
    plt.show()
    # logarithmic scale
