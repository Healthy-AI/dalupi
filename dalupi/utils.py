import os
import yaml
import torch
import copy
import datetime
import pickle
from pathlib import Path
from os.path import join

def save_pickle(data, path, filename):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(join(path, filename + '.pickle'), 'wb') as f:
        pickle.dump(data, f)

def save_yaml(data, path, filename):
    with open(join(path, filename + '.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def create_results_dir(config, suffix=None, update_config=False):
    time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    if suffix is not None:
        time_stamp += '_' + suffix
    results_path = join(config['results']['path'], time_stamp)
    Path(results_path).mkdir(parents=True, exist_ok=True)
    if update_config:
        config = copy.deepcopy(config)
        config['results']['path'] = results_path
        return results_path, config
    else:
        return results_path

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_yaml(path):
    with open(path) as file:
        yaml_file = yaml.safe_load(file)
    return yaml_file

def _change_to_local_paths(d, cluster_project_path, local_project_path):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            recursive = {
                k: _change_to_local_paths(
                    v,
                    cluster_project_path,
                    local_project_path
                )
            }
            out.update(recursive)
        elif isinstance(k, str) and 'path' in k and isinstance(v, str):
            out[k] = v.replace(
                cluster_project_path,
                local_project_path
            )
        else:
            out[k] = v
    return out

def load_config(config_path):
    config = load_yaml(config_path)
    try:
        local_home_path = os.environ['LOCAL_HOME_PATH']
        cluster_project_path = os.environ['CLUSTER_PROJECT_PATH']
        local_project_path = os.environ['LOCAL_PROJECT_PATH']
        home_path = str(Path.home())
        if home_path == local_home_path:
            return _change_to_local_paths(
                config,
                cluster_project_path,
                local_project_path
            )
    except KeyError:
        return config
