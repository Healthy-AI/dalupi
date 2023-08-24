import os
import copy
import random
import warnings
import torchvision
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import join
from .datasets import BaseDataset
from dalupi.models.transforms import ToTensor, RandomHorizontalFlip, Compose
from dalupi.data.utils import check_get_data_arguments

__ALL__ = [
    'Pipeline1',
    'Pipeline2',
    'Pipeline3',
    'Pipeline4',
]

class BasePipeline:
    def __init__(
        self,
        source_path,
        target_path,
        num_train_source,
        num_train_target,
        num_valid_source,
        num_valid_target,
        num_test_source,
        num_test_target,
        num_train_sample=None,
        seed=None
    ):
        self.source_path = source_path
        self.target_path = target_path

        self.csv_data_source = pd.read_csv(join(source_path, 'data.csv'))
        self.csv_pi_source = pd.read_csv(join(source_path, 'pi.csv'))

        self.csv_data_target = pd.read_csv(join(target_path, 'data.csv'))
        self.csv_pi_target = pd.read_csv(join(target_path, 'pi.csv')) 
        
        self.num_train_source = num_train_source
        self.num_train_target = num_train_target
        self.num_valid_source = num_valid_source
        self.num_valid_target = num_valid_target
        self.num_test_source = num_test_source
        self.num_test_target = num_test_target

        self.num_train_sample = num_train_sample
        
        self.seed = seed
    
    def describe_test_data(self, out_path):
        for domain in ['source', 'target']:
            test = getattr(self, '%s_test' % domain)
            BaseDataset((test, None)).describe(save_to=out_path, suffix='test_%s' % domain)
    
    def _compute_sampling_probabilities(self, dataset):
        raise NotImplementedError
    
    def _collect_weights(self, dataset, subset=None, num_finding=None, num_no_finding=None):
        raise NotImplementedError
    
    def _sample_data(self, dataset, subset, num_samples):
        num_finding, num_no_finding = num_samples
        weights_finding, weights_no_finding = self._collect_weights(dataset, subset=subset, num_finding=num_finding, num_no_finding=num_no_finding)
        csv_finding = dataset.csv.sample(n=num_finding, weights=weights_finding, random_state=self.seed, ignore_index=True)
        csv_no_finding = dataset.csv.sample(n=num_no_finding, weights=weights_no_finding, random_state=self.seed, ignore_index=True)
        csv = pd.concat((csv_finding, csv_no_finding), ignore_index=True)
        if len(csv) > 0:
            csv.loc[:, 'subset'] = subset
        return csv

    def _split_data(self, domain):
        '''
        Select data for training, validation, and testing.
        '''
        csv_data = getattr(self, 'csv_data_%s' % domain)
        csv_pi = getattr(self, 'csv_pi_%s' % domain)
        X = (csv_data, csv_pi)
        dataset = BaseDataset(X)
        dataset.domain = domain

        # Remove samples for which there are only findings that are not considered
        zero_indexes = dataset.csv[dataset.labels.sum(axis=1) == 0].imageindex
        dataset.remove(zero_indexes, remove_patients=False)

        # Collect training data with PI
        pi_indexes = dataset.csv_pi.imageindex
        train_pi = dataset.csv[dataset.csv.imageindex.isin(pi_indexes)].copy()
        train_pi.loc[:, 'subset'] = 'train'

        # Compute sampling probabilities
        self._compute_sampling_probabilities(dataset)

        # Make a copy of the dataset for sampling training data
        dataset_train = copy.deepcopy(dataset)
    
        # Sample test data
        dataset.remove(pi_indexes)
        dataset_train.remove(pi_indexes, remove_patients=False)
        n_test = getattr(self, 'num_test_%s' % domain)
        test = self._sample_data(dataset, 'test', n_test)
        assert len(test.imageindex) == len(set(test.imageindex))
        assert set(pi_indexes).isdisjoint(test.imageindex)

        # Sample validation data
        dataset.remove(test.imageindex)
        dataset_train.remove(test.imageindex)
        n_valid = getattr(self, 'num_valid_%s' % domain)
        valid = self._sample_data(dataset, 'valid', n_valid)
        assert len(valid.imageindex) == len(set(valid.imageindex))
        assert set(pi_indexes).isdisjoint(valid.imageindex)
        assert set(test.imageindex).isdisjoint(valid.imageindex)

        # Sample training data without PI
        dataset_train.remove(valid.imageindex)
        n_train = getattr(self, 'num_train_%s' % domain)
        train_nopi = self._sample_data(dataset_train, 'train', n_train)
        assert len(train_nopi.imageindex) == len(set(train_nopi.imageindex))
        assert set(pi_indexes).isdisjoint(train_nopi.imageindex)
        assert set(test.imageindex).isdisjoint(train_nopi.imageindex)
        assert set(valid.imageindex).isdisjoint(train_nopi.imageindex)

        # Set attributes
        setattr(self, '%s_dataset' % domain, dataset_train)
        setattr(self, '%s_train_nopi' % domain, train_nopi)
        setattr(self, '%s_train_pi' % domain, train_pi)
        setattr(self, '%s_valid' % domain, valid)
        setattr(self, '%s_test' % domain, test)

    def split_data(self):
        self._split_data(domain='source')
        self._split_data(domain='target')

    def _get_train_data(self, setting):
        # Start with samples for which PI is available
        if setting == 'source' or setting == 'lupi':
            csv_data_pi = self.source_train_pi
            num_finding = len(csv_data_pi)
        elif setting == 'target':
            csv_data_pi = self.target_train_pi
            num_finding = len(csv_data_pi)
        elif setting in {'dann', 'dalupi'} or setting.startswith('adapt'):
            csv_data_pi = pd.concat(
                (self.source_train_pi, self.target_train_pi),
                ignore_index=True,
                join='inner'
            )
            num_finding = len(self.source_train_pi)  # PI samples from target is considered bonus information

        # Add samples that do not contain PI
        domain = 'target' if setting == 'target' else 'source'
        if self.num_train_sample is not None:
            if isinstance(self.num_train_sample, (tuple, list)):
                num_finding_sample, num_no_finding_sample = self.num_train_sample
            else:
                num_finding_sample = self.num_train_sample
                num_no_finding_sample = 0
            num_finding_sample = max([0, num_finding_sample - num_finding])
            dataset = getattr(self, '%s_dataset' % domain)
            csv_data_nopi = self._sample_data(
                dataset,
                'train',
                (num_finding_sample, num_no_finding_sample)
            )
        else:
            csv_data_nopi = getattr(self, '%s_train_nopi' % domain)
        
        csv_data = pd.concat((csv_data_pi, csv_data_nopi), ignore_index=True)

        # Include PI information
        if setting in {'source', 'target', 'dann'} or setting.startswith('adapt'):
            csv_pi = None
        elif setting == 'lupi':
            csv_pi = self.csv_pi_source
        elif setting == 'dalupi':
            csv_pi = pd.concat(
                (self.csv_pi_source, self.csv_pi_target),
                ignore_index=True
            )
        
        return csv_data, csv_pi

    def _get_eval_data(self, subset, setting, domain):
        if subset == 'valid':
            csv_data = getattr(self, '%s_valid' % domain)
        else:
            csv_data = getattr(self, '%s_test' % domain)
        if setting in {'lupi', 'dalupi'}:
            csv_pi = getattr(self, 'csv_pi_%s' % domain)
        else:
            csv_pi = None
        return csv_data, csv_pi

    def get_data(self, subset, setting, valid_domain=None, prediction_domain=None):
        check_get_data_arguments(subset, setting, valid_domain, prediction_domain)
        
        if subset == 'train':
            csv_data, csv_pi = self._get_train_data(setting)
        elif subset == 'train_valid':
            csv_data_train, csv_pi = self._get_train_data(setting)
            csv_data_valid, _ = self._get_eval_data(subset='valid', setting=setting, domain=valid_domain)
            csv_data = pd.concat(
                (csv_data_train, csv_data_valid),
                ignore_index=True
            )
        else:
            csv_data, csv_pi = self._get_eval_data(subset, setting, prediction_domain)

        assert isinstance(csv_data.index, pd.RangeIndex)

        X = (csv_data, csv_pi)
        y = None

        return X, y
    
    def cleanup(self):
        pass

class Pipeline1(BasePipeline):
    '''
    Sample non-PI finding samples according to the distribution of
    the PI samples.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _compute_sampling_probabilities(self, dataset):
        pi_indexes = dataset.csv_pi.imageindex
        mask = dataset.csv.imageindex.isin(pi_indexes)
        combs = pd.DataFrame(
            dataset.labels,
            index=dataset.csv.index,
            columns=dataset.pathologies
        )[mask].value_counts(normalize=True)
        probs = combs.values
        combs = list(combs.index)
        dataset.probs = probs
        dataset.combs = combs

    def _collect_weights(self, dataset, num_finding, num_no_finding=None, subset=None):
        labels = pd.Series(map(tuple, dataset.labels))
        weights_finding = pd.Series(0, index=dataset.csv.index)
        for c, p in zip(dataset.combs, dataset.probs):
            mask = labels == c
            if mask.sum() == 0:
                raise ValueError('The dataset contains zero samples of class (%s).' % ', '.join(map(str, c)))
            if mask.sum() < num_finding * p:
                f = (num_finding * p, ', '.join(map(str, c)), p, mask.sum())
                warnings.warn('About %.0f samples are required from class (%s) to achive proportion %.2f but only %d samples are available.' % f)
            weights_finding[mask] = p / mask.sum()
        weights_nofinding = dataset.csv['No Finding']
        return weights_finding, weights_nofinding

class Pipeline2(Pipeline1):
    '''
    Sample non-PI finding samples according to the distribution of
    the PI samples. Double the number of PI samples by flipping all images 
    horizontally.
    '''
    def __init__(self, image_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path

    def _augment_data(self, domain):
        csv_data = getattr(self, 'csv_data_%s' % domain)
        csv_pi = getattr(self, 'csv_pi_%s' % domain)
        X = (csv_data, csv_pi)
        transforms = Compose([ToTensor(), RandomHorizontalFlip(p=1)])
        dataset = BaseDataset(X, views='*', transforms=transforms)

        pi_mask = csv_data.imageindex.isin(csv_pi.imageindex)
        indexes = np.arange(stop=len(csv_data))[pi_mask]

        augmented_rows = []
        augmented_rows_pi = []

        for i, j in enumerate(indexes):
            sample = dataset.getitem(j)
            imageindex = join(self.image_path, 'augmented_pi_images', '%s_%s.jpg' % (domain, str(i+1).zfill(4)))
            torchvision.utils.save_image(sample['img'], imageindex)
            
            row = csv_data.iloc[j].copy()
            row.imageindex = imageindex
            augmented_rows.append(row.to_frame().T)

            rows_pi = csv_pi[csv_pi.imageindex == csv_data.iloc[j].imageindex].copy()
            rows_pi.imageindex = imageindex
            x1y1x2y2 = sample['target']['boxes'].numpy()
            rows_pi.x = x1y1x2y2[:, 0]
            rows_pi.y = x1y1x2y2[:, 1]
            rows_pi.w = x1y1x2y2[:, 2] - x1y1x2y2[:, 0]  # x2 -> w
            rows_pi.h = x1y1x2y2[:, 3] - x1y1x2y2[:, 1]  # y2 -> h
            augmented_rows_pi.append(rows_pi)
        
        csv_data = pd.concat([csv_data] + augmented_rows, ignore_index=True)
        csv_pi = pd.concat([csv_pi] + augmented_rows_pi, ignore_index=True)

        setattr(self, 'csv_data_%s' % domain, csv_data)
        setattr(self, 'csv_pi_%s' % domain, csv_pi)
    
    def split_data(self):
        Path(join(self.image_path, 'augmented_pi_images')).mkdir(exist_ok=True)
        for domain in ['source', 'target']:
            self._augment_data(domain)
            self._split_data(domain)
    
    def cleanup(self):
        imagefolder = join(self.image_path, 'augmented_pi_images')
        os.system('rm -rf %s' % imagefolder)

class Pipeline3(BasePipeline):
    '''
    Sample non-PI finding samples randomly and do not differentiate
    between "finding" and "no finding" samples.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(self.seed)
    
    def _compute_sampling_probabilities(self, dataset):
        pass

    def _sample_data(self, dataset, subset, num_samples):
        if isinstance(num_samples, (tuple, list)):
            assert len(num_samples) == 2 and num_samples[-1] == 0
            num_samples = num_samples[0]
        num_sampled_images = 0
        unique_patients = set(dataset.csv.patientid.tolist())
        sampled_patients = []
        while num_sampled_images < num_samples:
            # Sample a random patient
            pid = random.sample(unique_patients, 1)[0]
            sampled_patients.append(pid)
            num_sampled_images += (dataset.csv.patientid == pid).sum()
            unique_patients.remove(pid)
        csv = dataset.csv[dataset.csv.patientid.isin(sampled_patients)]
        csv = csv.sample(num_samples, random_state=self.seed)
        if len(csv) > 0:
            csv.loc[:, 'subset'] = subset
        return csv.reset_index(drop=True)

class Pipeline4(Pipeline3, Pipeline2):
    '''
    Sample non-PI finding samples randomly and do not differentiate
    between "finding" and "no finding" samples. Double the number 
    of PI samples by flipping all images horizontally.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
