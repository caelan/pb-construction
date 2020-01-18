##################################################    

def export_log_data(extrusion_file_path, log_data, overwrite=True, indent=None):
    import os
    import datetime
    import json
    from collections import OrderedDict

    with open(extrusion_file_path, 'r') as f:
        shape_data = json.loads(f.read())
    
    if 'model_name' in shape_data:
        file_name = shape_data['model_name']
    else:
        file_name = extrusion_file_path.split('.json')[-2].split(os.sep)[-1]

    # result_file_dir = r'C:\Users\yijiangh\Documents\pb_ws\pychoreo\tests\test_data'
    here = os.path.abspath(os.path.dirname(__file__))
    result_file_dir = here
    result_file_dir = os.path.join(result_file_dir, 'extrusion_log')
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir) 
    
    data = OrderedDict()
    data['file_name'] = file_name
    date = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    data['write_time'] = date
    data.update(log_data)

    file_name_tag = log_data['search_method'] + '-' + log_data['heuristic']
    # if log_data['heuristic'] in ['stiffness', 'fixed-stiffness']:
    #     file_name_tag += '-' + log_data['stiffness_criteria']
    plan_path = os.path.join(result_file_dir, '{}_log_{}{}.json'.format(file_name, 
        file_name_tag,  '_'+data['write_time'] if not overwrite else ''))
    with open(plan_path, 'w') as f:
        json.dump(data, f, indent=indent)