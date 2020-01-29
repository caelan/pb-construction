import os
import datetime
import json
from collections import OrderedDict, namedtuple
import heapq
from termcolor import cprint
from pybullet_tools.utils import INF

from extrusion.stiffness import TRANS_TOL, ROT_TOL
from extrusion.utils import RESOLUTION
from extrusion.stream import STEP_SIZE, APPROACH_DISTANCE, MAX_DIRECTIONS, MAX_ATTEMPTS

MAX_BACKTACK = INF

# log options
RECORD_BT = True
RECORD_CONSTRAINT_VIOLATION = True

QUEUE_COUNT = 5 # number of candidates on the queue to be recorded at each iter
RECORD_QUEUE = False

OVERWRITE = True # add time tag if not overwrite

# visual diagnosis options
VISUALIZE_ACTION = False # visualize action step-by-step
CHECK_BACKTRACK = False # visually check

# video recording
RECORD_VIDEO = True


def get_global_parameters():
    return {
        'translation_tolerance': TRANS_TOL,
        'rotation_tolerance': ROT_TOL,
        'joint_resolution': RESOLUTION,
        'step_size': STEP_SIZE,
        'approach_distance': APPROACH_DISTANCE,
        'max_directions': MAX_DIRECTIONS,
        'max_attempts': MAX_ATTEMPTS,
    }

def config_specific_file_name(config, overwrite, tag=None, interfix='', suffix='.json'):
    date_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    file_name = '{}_{}_{}-{}{}{}{}'.format(config.problem, interfix,
        config.algorithm, config.bias, 
        '_'+tag if tag else '',
        '_'+date_time if not overwrite else '',
        suffix)
    return file_name

##################################################    

def export_log_data(extrusion_file_path, log_data, overwrite=True, indent=None, tag=None, \
    collisions=True, disable=False, stiffness=True, motions=True, lazy=False, **kwargs):
    file_name = extrusion_file_path.split('.json')[-2].split(os.sep)[-1]

    # result_file_dir = r'C:\Users\yijiangh\Documents\pb_ws\pychoreo\tests\test_data'
    here = os.path.abspath(os.path.dirname(__file__))
    result_file_dir = here
    result_file_dir = os.path.join(result_file_dir, 'extrusion_log')
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir) 
    
    data = OrderedDict()
    data['problem'] = file_name
    date = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    data['write_time'] = date
    data['parameters'] = get_global_parameters()

    # configs
    data['plan_extrusions'] = not disable
    data['use_collisions'] = collisions
    data['use_stiffness'] = stiffness
    data['lazy'] = lazy

    data.update(log_data)

    if stiffness and disable:
        tag = 'stiffness_only'

    Config = namedtuple('Config', ['problem', 'algorithm', 'bias'])
    config = Config(file_name, data['algorithm'], data['heuristic'])
    plan_path = os.path.join(result_file_dir, config_specific_file_name(config, 
        overwrite=overwrite, tag=tag, interfix='log'))
    # plan_path = os.path.join(result_file_dir, '{}_log_{}{}{}.json'.format(file_name, file_name_tag,
    #     '_'+tag if tag else '',
    #     '_'+data['write_time'] if not overwrite else ''))
    with open(plan_path, 'w') as f:
        json.dump(data, f, indent=indent)
    print('------')
    cprint('Log file saved to: {}'.format(plan_path), 'green')

##################################################    

def export_result_data(config, plan_data, overwrite=True, indent=None, tag=None):
    # result_file_dir = "C:/Users/yijiangh/Documents/pb_ws/pychoreo/tests/test_data"
    here = os.path.abspath(os.path.dirname(__file__))
    result_file_dir = os.path.join(here, 'extrusion_results')
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir) 

    plan_path = os.path.join(result_file_dir, config_specific_file_name(config, 
        overwrite=overwrite, tag=tag, interfix='result'))
    # plan_path = os.path.join(result_file_dir, '{}_result_{}-{}{}{}.json'.format(config.problem, \
    #     config.algorithm, config.bias, 
    #     '_'+tag if tag else '',
    #     '_'+date_time if not overwrite else ''))

    if 'safe' in plan_data:
        plan_data['safe'] = bool(plan_data['safe'])
    if 'valid' in plan_data:
        plan_data['valid'] = bool(plan_data['valid'])

    with open(plan_path, 'w') as f:
        json.dump(plan_data, f, indent=None)
    print('------')
    cprint('Result saved to: {}'.format(plan_path), 'green')

##################################################    

def export_video_path(config, tag=None):
    here = os.path.abspath(os.path.dirname(__file__))
    result_file_dir = os.path.join(here, 'extrusion_videos')
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir) 

    file_name = 'video.mp4' if config is None else config_specific_file_name(config, overwrite=OVERWRITE, interfix='video', suffix='.mp4')
    plan_path = os.path.join(result_file_dir, file_name)
    return plan_path