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

from extrusion.experiment import EXCLUDE #, EXPERIMENTS_DIR
from extrusion.parsing import get_extrusion_dir
from pddlstream.utils import INF, str_from_object, get_python_version
from pybullet_tools.utils import read_pickle, implies

OVERLEAF = '/Users/caelan/Desktop/Construction/20-01-31_14-54/'
RELATIVE = 'figures/regression-StiffPlan_result_repo/'

WIDTH = 0.32

# https://en.wikibooks.org/wiki/LaTeX/Floats,_Figures_and_Captions

# TEMPLATE = """
# \\begin{{subfigure}}[b]{{{width}\\textwidth}}
#     \includegraphics[width=\\textwidth]{{{path}}}
#     \caption{{{caption}}}
#     \label{{fig:{name}}}
# \end{{subfigure}}"""

# TEMPLATE = """
# \\begin{{figure*}}[ht]
#  \centering
#  \includegraphics[width={width}\\textwidth]{{{path}}}
#  \caption{{{caption}}}
#  \label{{fig:{name}}}
# \end{{figure*}}"""

TEMPLATE = "\includegraphics[width={width}\\textwidth]{{{path}}}"

"""
 problem: {C_shape, bunny_full_quad, bunny_full_tri, bunny_full_tri_dense, 
 compas_fea_beam_tree, compas_fea_beam_tree_M, compas_fea_beam_tree_M_simp, 
 compas_fea_beam_tree_S, compas_fea_beam_tree_S_simp, compas_fea_truss_frame, 
 david, dented_cube, djmm_test_block_S1_03-14-2019_w_layer, duck, extreme_beam_test, 
 extrusion_exp_L75.0, fandisk, fertility, four-frame, klein_bottle, klein_bottle_S1.5, 
 klein_bottle_trail_S1.5, klein_bottle_trail_S2, long_beam_test, mars_bubble_S1_03-14-2019_w_layer, 
 robarch_tree_M, robarch_tree_S, semi_sphere, sig_artopt-bunny_S1_03-14-2019_w_layer, simple_frame, 
 topopt-100_S1_03-14-2019_w_layer, topopt-101_tiny, topopt-205_S0.7_03-14-2019_w_layer, 
 topopt-205_long_beam_test, topopt-205_rotated, topopt-205_rotated_S1.35, topopt-205_rotated_S1.5, 
 topopt-310_S1_03-14-2019_w_layer, tre_foil_knot, tre_foil_knot_S1.35, voronoi_S1_03-14-2019_w_layer}
"""

def main():
    #assert get_python_version() == 3
    parser = argparse.ArgumentParser()
    # parser.add_argument('path', help='Analyze an experiment')
    # parser.add_argument('-a', '--all', action='store_true',
    #                     help='Enables the viewer during planning')
    # parser.add_argument('-d', '--dump', action='store_true',
    #                     help='Dumps the configuration for each ')
    args = parser.parse_args()
    #np.set_printoptions(precision=3)


    instances = set(os.path.splitext(filename)[0] for filename in os.listdir(get_extrusion_dir())
                    if os.path.splitext(filename)[1] == '.json')
    filenames = set(os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(OVERLEAF, RELATIVE)))
    print('Missing images:', sorted(instances - filenames))
    print('Missing instances:', sorted(filenames - instances))
    print(5*'\n')

    printed = set()
    for filename in sorted(os.listdir(os.path.join(OVERLEAF, RELATIVE))):
        name, ext = os.path.splitext(filename)
        assert ext == '.png'
        if name in EXCLUDE:
            continue
        path = os.path.join(RELATIVE, filename)
        printed.add(name)
        #name = name.lower()
        #path = os.path.join()
        if len(printed) % (5*3+1) == 0:
            print()
        print(TEMPLATE.format(width=WIDTH, path=path, caption=name.replace('_', '\_'), name=name))

    print(5*'\n')
    print(len(printed), sorted(printed))

if __name__ == '__main__':
    main()
