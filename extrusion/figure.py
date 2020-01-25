#!/usr/bin/env python

from __future__ import print_function

import math
import scipy.stats

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
#from tabulate import tabulate

from extrusion.experiment import EXCLUDE, Configuration, EXPERIMENTS_DIR, HEURISTICS, ALGORITHMS

SUCCESS = 'success'
RUNTIME = 'runtime'
SCORES = [SUCCESS, RUNTIME]

FONT_SIZE = 14
WIDTH = 0.2
ALPHA = 1.0 # 0.5

##################################################

RENAME_LABELS = {
    'none': 'random',
    'z': 'task-distance',
    'dijkstra': 'truss-distance',
    'plan-stiffness': 'stiffness-plan',
    'lookahead': 'progression+lookahead',
    SUCCESS: '% solved',
    RUNTIME: 'runtime (sec)',
}

def rename(name):
    return RENAME_LABELS.get(name, name)

##################################################

def bar_graph(data, attribute):
    matplotlib.rcParams.update({'font.size': FONT_SIZE})
    #pltfig, ax = plt.subplots()
    hatch = '/' if attribute == RUNTIME else None
    ax = plt.subplot()
    for h_idx, heuristic in enumerate(HEURISTICS): # Add everything with the same label at once
        algorithms = []
        values = []
        for algorithm in ALGORITHMS:
            key = frozenset({'algorithm': algorithm, 'bias': heuristic}.items())
            if key in data:
                algorithms.append(rename(algorithm))
                values.append(data[key][attribute])
        if not algorithms:
            continue
        indices = np.array(range(len(algorithms)))
        means = list(map(np.mean, values)) # 100
        stds = list(map(np.std, values))
        rects = plt.bar(h_idx*WIDTH + indices, means, WIDTH, alpha=ALPHA, hatch=hatch,
                        label=rename(heuristic)) # align='center'
        #for rect in rects:
        #    h = rect.get_height()
        #    ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h), ha='center', va='bottom')
        # TODO: confidence intervals for runtime

    plt.title('Stiffness')
    ticks = np.arange(len(ALGORITHMS)) + len(ALGORITHMS)/2.*WIDTH
    plt.xticks(ticks, map(rename, ALGORITHMS))
    plt.xlabel('Algorithm')
    ax.autoscale(tight=True)
    plt.legend(loc='upper left')
    plt.ylabel(rename(attribute))
    #plt.savefig('test')
    plt.tight_layout()
    #plt.grid()
    plt.show()
