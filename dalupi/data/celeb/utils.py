import torch
from copy import deepcopy
import importlib
import inspect
from torchvision import transforms
from sklearn.model_selection import train_test_split

# https://github.com/p-lambda/in-n-out/blob/main/innout/datasets/celeba.py#L28
def resize_transform():
    return transforms.Compose([
        transforms.Resize((64,64), interpolation=2),
        transforms.ToTensor()])

# https://github.com/p-lambda/in-n-out/blob/main/innout/datasets/celeba.py#L34
def tensor_transform():
    return transforms.ToTensor()

# https://github.com/p-lambda/in-n-out/blob/main/innout/load_utils.py#L11
def initialize(obj_config, update_args=None):
    classname = obj_config['classname']
    kwargs = obj_config.get('args')
    if kwargs is None:
        kwargs = {}
    if update_args is not None:
        kwargs.update(update_args)
    return initialize_obj(classname, kwargs)

# https://github.com/p-lambda/in-n-out/blob/main/innout/load_utils.py#L54
def initialize_obj(classname, args_dict=None):
    module_name, class_name = classname.rsplit(".", 1)
    Class = getattr(importlib.import_module(module_name), class_name)
    # filter by argnames
    if args_dict is not None:
        argspec = inspect.getfullargspec(Class.__init__)
        argnames = argspec.args
        args_dict = {k: v for k, v in args_dict.items()
                     if k in argnames or argspec.varkw is not None}

        defaults = argspec.defaults
        # add defaults
        if defaults is not None:
            for argname, default in zip(argnames[-len(defaults):], defaults):
                if argname not in args_dict:
                    args_dict[argname] = default
        class_instance = Class(**args_dict)
    else:
        class_instance = Class()
    return class_instance

# https://github.com/p-lambda/in-n-out/blob/main/innout/load_utils.py#L102
def init_transform(config, transform_type):
    '''
    Initializes a PyTorch transform from a config file.

    Parameters
    ----------
    config : dict
        Dictionary representation of .yaml config file.
    transform_type: str
        One of 'train[_target]', 'eval_train[_target]', 'val[_target]', or
        'test[_target]'.

    Returns
    -------
    torchvision.Transform
    '''
    if transform_type + '_transforms' not in config:
        return None

    config_transforms = config[transform_type + '_transforms']
    transform_list = [initialize(trans) for trans in config_transforms]
    return transforms.Compose(transform_list)

# https://github.com/p-lambda/in-n-out/blob/main/innout/load_utils.py#L126
def init_dataset(config, dataset_type, template_dataset=None):
    '''
    Initializes a PyTorch Dataset for train, eval_train, validation, or test.

    A few notes:
        - 'train' and 'eval_train' use 'train_transforms'.
        - 'val' and 'test' use 'test_transforms'.
        - 'eval_train' defaults to args in 'train_args' and then updates using
          args in 'eval_train_args'.
        - if config['dataset']['args']['standardize'] is True, then
            - if dataset_type is 'train', then the mean/std of the training
              set is saved in the config file after loading.
            - otherwise, the saved mean/std of the training set from the config
              is used to overwrite the mean/std of the current Dataset.
          Hence, it's important for standardization that the training set is
          first loaded before eval_train/val/test.

    Parameters
    ----------
    config : dict
        Dictionary representation of .yaml config file.
    dataset_type : str
        Either 'train', 'eval-train', 'val', or 'test'.
    template_dataset : torch.utils.data.Dataset, default None
        Optional Dataset to use for initialization.

    Returns
    -------
    torch.utils.data.Dataset
    '''
    custom_type = False
    if dataset_type not in ['train', 'eval_train', 'val', 'test', 'test2']:
        custom_type = True

    transform_type = dataset_type
    if dataset_type in {'eval_train', 'val', 'test2'} or custom_type:
        transform_type = 'test'  # Use test transforms for eval sets.
    transform = init_transform(config, transform_type)
    target_transform = init_transform(config, transform_type + '_target')

    split_type = dataset_type
    if dataset_type == 'eval_train':
        split_type = 'train'  # Default eval_train split is 'train'.
    dataset_kwargs = {'split': split_type, 'transform': transform,
                      'target_transform': target_transform,
                      'template_dataset': template_dataset,
                      'eval_mode': (dataset_type != 'train')}

    if dataset_type == 'eval_train':  # Start off with args in 'train_args'.
        dataset_kwargs.update(config['dataset'].get('train_args', {}))
    dataset_kwargs.update(config['dataset'].get(dataset_type + '_args', {}))

    # We make a copy since the initialize function calls dict.update().
    dataset_config = deepcopy(config['dataset'])
    dataset = initialize(dataset_config, dataset_kwargs)

    if config['dataset'].get('args', {}).get('standardize'):
        if dataset_type == 'train':  # Save training set's mean/std.
            config['dataset']['mean'] = dataset.get_mean()
            config['dataset']['std'] = dataset.get_std()
        else:  # Update dataset with training set's mean and std.
            dataset.set_mean(config['dataset']['mean'])
            dataset.set_std(config['dataset']['std'])

    if config['dataset'].get('args', {}).get('standardize_output'):
        if dataset_type == 'train':  # Save training set's output mean/std.
            config['dataset']['output_mean'] = dataset.get_output_mean()
            config['dataset']['output_std'] = dataset.get_output_std()
        else:  # Update dataset with training set's output mean and std.
            dataset.set_output_mean(config['dataset']['output_mean'])
            dataset.set_output_std(config['dataset']['output_std'])

    return dataset

def _get_innout_data(innout_config, dataset_type):
    dataset = init_dataset(innout_config, dataset_type)
    dataset._only_unlabeled = False  # Always return targets.
    x = [d['data'] for d in iter(dataset)]
    w = [d['domain_label']['meta'].float() for d in iter(dataset)]
    y = [d['target'] for d in iter(dataset)]
    return torch.stack(x), torch.stack(w), torch.cat(y).long()

def _get_target_train_val(config, subset='train'):
    X, w, y = _get_innout_data(config['data']['innout'], 'out_unlabeled')
    test_size = config['data']['num_out_val']
    seed = config['data']['seed']
    X1, X2, w1, w2, y1, y2 = train_test_split(X, w, y, test_size=test_size, random_state=seed)
    if subset == 'train':
        return X1, w1, y1
    elif subset == 'val':
        return X2, w2, y2
    else:
        raise ValueError("Unknown subset %s." % subset)

def get_celeb_train_data(config, setting, use_source_pi=False, **kwargs):
    """Get data for model training."""

    innout_config = config['data']['innout']
    
    if setting == 'dalupi':
        if use_source_pi:
            # Get extra (X, w) source data for the x2w model. We should never
            # use these targets, so we set them to `None` to be safe.
            Xs, ws, ys = _get_innout_data(innout_config, 'in_unlabeled')
            ys = None
        else:
            Xs, ws, ys = _get_innout_data(innout_config, 'train')
        Xt, wt, _yt = _get_target_train_val(config, 'train')
        X = [Xs, Xt, ws, wt]
        y = ys
    elif setting.startswith('adapt'):
        Xs, _ws, ys = _get_innout_data(innout_config, 'train')
        Xt, _wt, _yt = _get_target_train_val(config, 'train')
        Xs_val, ys_val = get_celeb_val_data(config, setting)
        X = [Xs, Xt, Xs_val]
        y = [ys, ys_val]
    elif setting == 'source':
        X, _w, y = _get_innout_data(innout_config, 'train')
    elif setting == 'target':
        X, _w, y = _get_target_train_val(config, 'train')
    else:
        raise ValueError("Unknown setting %s." % setting)
    
    return X, y

def get_celeb_val_data(config, setting):
    """Get data for model validation during training."""

    innout_config = config['data']['innout']

    if setting == 'dalupi':
        Xs, ws, ys = _get_innout_data(innout_config, 'val')
        Xt, wt, _yt = _get_target_train_val(config, 'val')
        X = [Xs, Xt, ws, wt]
        y = ys
    elif setting == 'target':
        X, _w, y = _get_target_train_val(config, 'val')
    elif setting == 'source' or setting.startswith('adapt'):
        X, _w, y = _get_innout_data(innout_config, 'val')
    else:
        raise ValueError("Unknown setting %s." % setting)
    
    return X, y

def get_celeb_eval_data(config, setting, subset, prediction_domain):
    """Get data for evaluation of trained models."""

    assert subset in ['valid', 'test']

    innout_config = config['data']['innout']

    if prediction_domain == 'source':
        dataset_type = 'val' if subset == 'valid' else 'test'
        X, _w, y = _get_innout_data(innout_config, dataset_type)
    elif prediction_domain == 'target':
        if subset == 'valid':
            assert setting == 'target'
            X, _w, y = _get_target_train_val(config, 'val')
        else:
            X, _w, y = _get_innout_data(innout_config, 'test2')
    else:
        raise ValueError("Unknown prediction_domain %s." % prediction_domain)

    return X, y
    