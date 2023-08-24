import os
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from dalupi.data.utils import format_adapt_data
from sklearn.preprocessing import LabelBinarizer
from .prepare_data import make_dataset

def load_or_make_dataset(config):
    skew = config['data']['skew']
    seed = config['seed']  # we use this seed to make the data; config['data']['seed'] is for data splitting
    scale = config['data']['scale']
    
    if scale:
        file_name = 'MNIST_%0.1f_%d_scaled.pkl' % (skew, seed)
    else:
        file_name = 'MNIST_%0.1f_%d.pkl' % (skew, seed)
    
    path = os.path.join(config['data']['path'], file_name)
    
    if os.path.exists(path):
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
    else:
        dataset = make_dataset(skew, seed, scale, root=config['data']['path'])
        with open(path, 'wb') as handle:
            pickle.dump(dataset, handle)
    
    return dataset

def split_adapt_data(Xs, Xt, ys, test_size, seed):
    if isinstance(Xs, list):
        Xs = np.stack(Xs)
    if isinstance(Xt, list):
        Xt = np.stack(Xt)
    if isinstance(ys, list):
        ys = np.array(ys)

    # Convert to "channels last" format
    Xs = np.transpose(Xs, (0, 2, 3, 1))
    Xt = np.transpose(Xt, (0, 2, 3, 1))
    
    # Binarize labels
    ys = LabelBinarizer().fit_transform(ys)
    
    Xs_train, Xs_test, Xt_train, _, ys_train, ys_test = train_test_split(
        Xs, Xt, ys,
        test_size=test_size,
        random_state=seed,
    )

    X_train = np.concatenate([Xs_train, Xt_train])

    return X_train, ys_train, Xs_test, ys_test

def get_train_data(config, setting, **kwargs):
    x_source, y_source, x_target, y_target = load_or_make_dataset(config)

    w_source = [y[0] for y in y_source]
    y_source_mnist = [y[1] for y in y_source]

    w_target = [y[0] for y in y_target]
    y_target_mnist = [y[1] for y in y_target]

    test_size = config['data']['test_size']
    seed = config['data']['seed']
        
    if setting == 'dalupi':
        x_source = torch.stack(x_source)
        w_source = torch.stack(w_source)
        y_source_mnist = torch.tensor(y_source_mnist)

        x_source, _, w_source, _, y_source_mnist, _ = train_test_split(
            x_source, w_source, y_source_mnist,
            test_size=test_size,
            random_state=seed,
        )

        x_target = torch.stack(x_target)
        w_target = torch.stack(w_target)

        x_target, _, w_target, _ = train_test_split(
            x_target, w_target,
            test_size=test_size,
            random_state=seed,
        )

        X = [x_source, x_target, w_source, w_target]
        y = y_source_mnist
    
    elif setting.startswith('adapt'):
        X, y, _, _ = split_adapt_data(x_source, x_target, y_source_mnist, test_size, seed)

    elif setting == 'dann':
        X = x_source + x_target
        X = torch.stack(X)

        # The length of the labels must equal the length
        # of the inputs, but the target labels are not
        # used in the DANN network.
        y_mnist = y_source_mnist + y_target_mnist
        y_domain = np.concatenate((np.zeros_like(y_source_mnist), np.ones_like(y_target_mnist)))
        y = [[y_m, y_d] for y_m, y_d in zip(y_mnist, y_domain)]
        y = torch.tensor(y)

        X, _, y, _ = train_test_split(X, y, test_size=test_size, random_state=seed)

    elif setting == 'source':
        X = torch.stack(x_source)
        y = torch.tensor(y_source_mnist)
        X, _, y, _ = train_test_split(X, y, test_size=test_size, random_state=seed)

    else:
        X = torch.stack(x_target)
        y = torch.tensor(y_target_mnist)
        X, _, y, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
        
    return X, y

def get_eval_data(config, setting, subset, prediction_domain, get_split_datasets):
    assert subset in ['valid', 'test']

    x_source, y_source, x_target, y_target = load_or_make_dataset(config)

    if prediction_domain == 'target':
        X = x_target
        y = y_target
    else:
        X = x_source
        y = y_source
    
    X = np.stack(X) if setting.startswith('adapt') else torch.stack(X)
    
    y = [yi[1] for yi in y]  # MNIST labels
    y = np.array(y) if setting.startswith('adapt') else torch.tensor(y)

    test_size = config['data']['test_size']
    seed = config['data']['seed']

    if setting.startswith('adapt') and subset == 'valid':
        # Note: UDA models use validation data from the source domain
        # We obtain validation data by splitting the training data in format_adapt_data
        X, y, _, _ = split_adapt_data(x_source, x_target, y, test_size, seed)
        X, y = format_adapt_data(X, y, config, train=True)[-1]
        return X, y
    
    elif setting.startswith('adapt') and subset == 'test':
        # This may look strange but what happens is that we obtain
        # the test part of (X, y) (x_target is ignored)
        _, _, X, y = split_adapt_data(X, x_target, y, test_size, seed)
        return X, y

    elif subset == 'valid':
        X, _, y, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
        datasets = get_split_datasets(X, y)
        X, y = [], []
        for Xi, yi in datasets[1]:  # index 1 gives the validation set
            X.append(Xi)
            y.append(yi)
        X = torch.stack(X)
        y = torch.stack(y)
        return X, y
    
    elif subset == 'test':
        _, X, _, y = train_test_split(X, y, test_size=test_size, random_state=seed)
        return X, y
