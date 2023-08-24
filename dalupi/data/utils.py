import torch
import torchvision
import pydoc
import sklearn
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from .datasets import DANNDataset
from amhelpers.config_parsing import _check_value
from amhelpers.amhelpers import create_object_from_dict
from dalupi import ALL_SETTINGS
from PIL import Image
from functools import partial
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input

def create_dataset_from_config(config, *args):
    '''
    This function can be used to obtain the correct
    class labels from a DataFrame.
    '''
    dataset_type = config['default']['dataset']['type']
    dataset = pydoc.locate(dataset_type)(*args)
    return dataset

def adapt_generator(dataset, image_size, only_X=False):
    dataset.csv.reset_index(drop=True, inplace=True)
    
    for i, imageindex in dataset.csv.imageindex.items():
        image = Image.open(imageindex)
        image = image.resize(image_size, Image.BILINEAR)
        
        X = np.array(image, dtype='float32')

        if X.ndim == 3 and X.shape[-1] == 3:
            pass
        else:
            if X.ndim > 2:
                X = X[..., 0]
            X = X[..., np.newaxis]
            X = np.repeat(X, 3, -1)
        
        if only_X:
            yield preprocess_input(X)
        else:
            y = dataset.labels[i]
            yield (preprocess_input(X), y)

def format_adapt_data(X, y, config, train=True):
    experiment = config['experiment']

    if experiment == 'chestxray' or experiment == 'coco':
        image_size = (config['image_w'], config['image_h'])
        n_channels = config['image_c']
        n_classes = config['n_classes']
        
        assert n_channels == 3
        
        if train:
            X = X[0]
            
            Xys_train = X[(X.domain == 'source') & (X.subset == 'train')]
            Xys_train = Xys_train.sample(frac=1, random_state=config['data']['seed'])
            Xys_train = create_dataset_from_config(config, (Xys_train, None))
            
            Xt_train = X[(X.domain == 'target') & (X.subset == 'train')]
            Xt_train = Xt_train.sample(frac=1, random_state=config['data']['seed'])
            Xt_train = create_dataset_from_config(config, (Xt_train, None))
            
            Xys_valid = X[(X.subset == 'valid')]
            Xys_valid = create_dataset_from_config(config, (Xys_valid, None))

            Xys_train = tf.data.Dataset.from_generator(
                generator=partial(adapt_generator, Xys_train, image_size),
                output_types=(tf.float32, tf.float32),
                output_shapes=([*image_size, n_channels], [n_classes])
            )
            
            Xt_train = tf.data.Dataset.from_generator(
                generator=partial(adapt_generator, Xt_train, image_size),
                output_types=tf.float32,
                output_shapes=([*image_size, n_channels]),
                args=(True,),
            )
            
            Xys_valid = tf.data.Dataset.from_generator(
                generator=partial(adapt_generator, Xys_valid, image_size),
                output_types=(tf.float32, tf.float32),
                output_shapes=([*image_size, n_channels], [n_classes])
            )
        
            return Xys_train, None, Xt_train, Xys_valid
        
        else:
            X_test = X[0]
            X_test = create_dataset_from_config(config, (X_test, None))

            y_test = X_test.labels

            X_test = tf.data.Dataset.from_generator(
                generator=partial(adapt_generator, X_test, image_size),
                output_types=tf.float32,
                output_shapes=([*image_size, n_channels]),
                args=(True,),
            )
            
            return X_test, y_test

    elif experiment == 'mnist':
        if train:
            Xs_train, Xt_train = np.split(X, 2)
            ys_train = y
            
            valid_size = config['data']['valid_size']
            seed = config['data']['seed']
            Xs_train, Xs_val, ys_train, ys_val = train_test_split(
                Xs_train,
                ys_train,
                test_size=valid_size,
                random_state=seed
            )
            Xt_train, _ = train_test_split(Xt_train, test_size=valid_size, random_state=seed)
            
            return Xs_train, ys_train, Xt_train, (Xs_val, ys_val)
        
        else:
            return X, y

def check_get_data_arguments(subset, setting, valid_domain, prediction_domain):
    if not subset in {'train', 'train_valid', 'valid', 'test'}:
        raise ValueError("subset must be either 'train', 'valid', 'train_valid' or 'test'.")
    
    valid_settings = ALL_SETTINGS
    if not setting in valid_settings:
        raise ValueError('setting must be in {}.'.format(valid_settings))
    
    if subset == 'train_valid':
        if not valid_domain in {'source', 'target'}:
            raise ValueError("valid_domain must be either 'source' or 'target'.")

    if subset == 'valid' or subset == 'test':
        if not prediction_domain in {'source', 'target'}:
            raise ValueError("prediction_domain must be either 'source' or 'target'.")

def collate_fn(batch):
    images, targets = [], []
    for x, y in batch:
        images += [x]
        targets += [y]
    return images, targets

class CVSplitter:
    def __init__(self, dataset, dataset_params, **kwargs):
        self.dataset = dataset
        self.dataset_params = dataset_params
        self.splitter = GroupSplit(**kwargs)

    def split(self, X, y, groups=None):
        dataset = self.dataset(X, y, **self.dataset_params)
        return self.splitter.split(dataset)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitter.n_splits

class BaseSplit:
    def __init__(self, n_splits=1, train_size=0.8, random_state=None):
        self.n_splits = n_splits
        self.train_size = train_size
        self.random_state = random_state if random_state is not None else np.random.randint(0, 101)
    
    def split(self, dataset):
        raise NotImplementedError

    def __call__(self, dataset):
        ii_train, ii_valid = next(self.split(dataset))
        dataset_train = torch.utils.data.Subset(dataset, ii_train)
        dataset_valid = torch.utils.data.Subset(dataset, ii_valid)
        return dataset_train, dataset_valid

class StratifiedSplit(BaseSplit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def split(self, dataset):
        splitter = StratifiedShuffleSplit(n_splits=self.n_splits, train_size=self.train_size, random_state=self.random_state)
        y = dataset.labels
        args = (np.zeros(len(dataset)), y)
        return splitter.split(*args)

class GroupSplit(BaseSplit):
    def __init__(self, c_group, **kwargs):
        self.c_group = c_group
        super().__init__(**kwargs)
    
    def split(self, dataset):
        splitter = GroupShuffleSplit(n_splits=self.n_splits, train_size=self.train_size, random_state=self.random_state)
        groups = dataset.csv[self.c_group]
        args = (np.zeros(len(dataset)), None, groups)
        yield from splitter.split(*args)

class TrainValidSplit:
    def __init__(self, num_valid_pi=0, pi_splitter=None, random_state=None):
        self.num_valid_pi = num_valid_pi
        self.pi_splitter = pi_splitter
        self.random_state = random_state
    
    def __call__(self, dataset):
        split_generator = self.split(dataset)
        _, ii_train = next(split_generator)
        _, ii_valid = next(split_generator)
        dataset_train = torch.utils.data.Subset(dataset, ii_train)
        dataset_valid = torch.utils.data.Subset(dataset, ii_valid)
        return dataset_train, dataset_valid

    def split(self, dataset):
        assert isinstance(dataset.csv.index, pd.RangeIndex)

        ii_valid = dataset.csv[dataset.csv.subset == 'valid'].index.tolist()

        if self.num_valid_pi > 0:
            pi_splitter = self.pi_splitter(self.num_valid_pi, self.random_state)
            _, ii_valid_pi = next(pi_splitter.split(dataset))
            ii_valid.extend(ii_valid_pi)

        # Avoid taking len(dataset) since length of DANN dataset is not the full length
        test_fold = np.zeros(len(dataset.labels))
        test_fold[ii_valid] = 1

        dataset.csv.loc[:, 'valid_split'] = test_fold.astype(bool)

        if isinstance(dataset, DANNDataset):
            # We should only iterate over the source samples
            ii_train_target = dataset.csv[
                (dataset.csv.subset == 'train') & (dataset.csv.domain == 'target')
            ].index.tolist()
            test_fold[ii_train_target] = -1

        splitter = sklearn.model_selection.PredefinedSplit(test_fold)
        yield from splitter.split()

def get_pipeline(config):
    pipeline_dict = config['data']['pipeline']
    pipeline_dict.pop('is_called', None)
    pipeline_params = {k: _check_value(v) if not k == 'type' else v for k, v in pipeline_dict.items()}
    pipeline = pipeline_params['type']
    if pipeline.endswith('chestxray.Pipeline2'):
        pipeline_params['image_path'] = config['results']['path']
    if pipeline.endswith('chestxray.Pipeline4'):
        pipeline_params['image_path'] = config['results']['path']
    return create_object_from_dict(pipeline_params)

def norm_flat_image(image, size=None):
    n_channels = image.shape[0]
    assert n_channels in [1, 3]

    # ImageNet statistics
    inv_std = [1 / 0.226] if n_channels == 1 else [1 / 0.229, 1 / 0.224, 1 / 0.225]
    inv_mean = [-0.449] if n_channels == 1 else [-0.485, -0.456, -0.406]
    
    inverse_transforms = [
        torchvision.transforms.Normalize(mean=n_channels * [0.], std=inv_std),
        torchvision.transforms.Normalize(mean=inv_mean, std=n_channels * [1.]),
    ]
    if size is not None:
        inverse_transforms += [torchvision.transforms.Resize(size)]
    inverse_transform = torchvision.transforms.Compose(inverse_transforms)
    image = inverse_transform(image)
    
    image = image.numpy()
    if n_channels == 3:
        image = image[0, :, :] + image[:, :, 1] + image[:, :, 2]
    else:
        image = image[0]
    return (image - np.min(image)) / (np.max(image) - np.min(image))
