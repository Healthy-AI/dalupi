import pandas as pd
import numpy as np
from os.path import join
from .datasets import BaseDataset
from sklearn.model_selection import train_test_split
from dalupi.data.utils import check_get_data_arguments

__ALL__ = ['Pipeline1']

class BasePipeline:
    def __init__(
        self,
        csv_path,
        num_train_source,
        num_train_target,
        num_valid_source,
        num_valid_target,
        num_test_source,
        num_test_target,
        seed=None
    ):
        self.csv_path = csv_path

        self.csv_data_source = pd.read_csv(join(csv_path, 'data_source.csv'))
        self.csv_pi_source = pd.read_csv(join(csv_path, 'pi_source.csv'))

        self.csv_data_target = pd.read_csv(join(csv_path, 'data_target.csv'))
        self.csv_pi_target = pd.read_csv(join(csv_path, 'pi_target.csv')) 
        
        self.num_train_source = num_train_source
        self.num_train_target = num_train_target
        self.num_valid_source = num_valid_source
        self.num_valid_target = num_valid_target
        self.num_test_source = num_test_source
        self.num_test_target = num_test_target
        
        self.seed = seed
    
    def _split_data(self, domain):
        data = getattr(self, 'csv_data_%s' % domain)
        pi = getattr(self, 'csv_pi_%s' % domain)
        
        data_train, data_test = train_test_split(
            data,
            train_size=getattr(self, 'num_train_%s' % domain),
            random_state=self.seed,
            shuffle=True
        )

        data_test, data_valid = train_test_split(
            data_test,
            train_size=getattr(self, 'num_test_%s' % domain),
            test_size=getattr(self, 'num_valid_%s' % domain),
            random_state=self.seed,
            shuffle=True
        )

        data_train.reset_index(drop=True, inplace=True)
        data_valid.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        data_train.loc[:, 'subset'] = 'train'
        data_valid.loc[:, 'subset'] = 'valid'
        data_test.loc[:, 'subset'] = 'test'

        setattr(self, 'data_train_%s' % domain, data_train)
        setattr(self, 'data_valid_%s' % domain, data_valid)
        setattr(self, 'data_test_%s' % domain, data_test)

        pi_train = pi[pi.imageindex.isin(data_train.imageindex)]
        pi_valid = pi[pi.imageindex.isin(data_valid.imageindex)]
        pi_test = pi[pi.imageindex.isin(data_test.imageindex)]

        setattr(self, 'pi_train_%s' % domain, pi_train)
        setattr(self, 'pi_valid_%s' % domain, pi_valid)
        setattr(self, 'pi_test_%s' % domain, pi_test)
    
    def split_data(self):
        self._split_data(domain='source')
        self._split_data(domain='target')
        self.data_train = pd.concat(
            [self.data_train_source, self.data_train_target],
            ignore_index=True
        )
    
    def describe_test_data(self, out_path):
        for domain in ['source', 'target']:
            test = getattr(self, 'data_test_%s' % domain)
            BaseDataset((test, None)).describe(save_to=out_path, suffix='test_%s' % domain)
    
    def _get_train_data(self, setting):
        raise NotImplementedError
    
    def _get_eval_data(self, subset, setting, domain):
        if subset == 'valid':
            csv_data = getattr(self, 'data_valid_%s' % domain)
            csv_pi = getattr(self, 'pi_valid_%s' % domain) if setting in {'lupi', 'dalupi'} else None
        else:
            csv_data = getattr(self, 'data_test_%s' % domain)
            csv_pi = getattr(self, 'pi_test_%s' % domain) if setting in {'lupi', 'dalupi'} else None
        return csv_data, csv_pi
    
    def get_data(self, subset, setting, valid_domain=None, prediction_domain=None):
        check_get_data_arguments(subset, setting, valid_domain, prediction_domain)
            
        if subset == 'train':
            csv_data, csv_pi = self._get_train_data(setting)
        elif subset == 'train_valid':
            csv_data_train, csv_pi_train = self._get_train_data(setting)
            csv_data_valid, csv_pi_valid = self._get_eval_data(subset='valid', setting=setting, domain=valid_domain)
            csv_data = pd.concat((csv_data_train, csv_data_valid), ignore_index=True)
            if csv_pi_valid is not None:
                csv_pi = pd.concat((csv_pi_train, csv_pi_valid), ignore_index=True)
            else:
                csv_pi = csv_pi_train
        else:
            csv_data, csv_pi = self._get_eval_data(subset, setting, prediction_domain)

        assert isinstance(csv_data.index, pd.RangeIndex)

        X = (csv_data, csv_pi)
        y = None

        return X, y
    
    def cleanup(self):
        pass

def _sample(data, size):
    if isinstance(size, float):
        assert size >= 0 and size <= 1
        size = int(size * len(data))
    return np.random.choice(data, size, replace=False)

class Pipeline1(BasePipeline):
    def __init__(self, num_train_pi_source, num_train_pi_target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_train_pi_source = num_train_pi_source
        self.num_train_pi_target = num_train_pi_target
    
    def _sample_pi(self, domain):
        csv_pi = getattr(self, 'pi_train_%s' % domain)
        indexes = csv_pi.imageindex.unique()
        num_pi = getattr(self, 'num_train_pi_%s' % domain)
        index_sample = _sample(indexes, num_pi)
        return csv_pi[csv_pi.imageindex.isin(index_sample)]
   
    def _get_train_data(self, setting):
        np.random.seed(self.seed)

        if setting in {'source', 'target'}:
            csv_data = getattr(self, 'data_train_%s' % setting)
            csv_pi = None
        elif setting == 'lupi':
            csv_data = getattr(self, 'data_train_source')
            csv_pi = self._sample_pi('source')
        elif setting in {'dalupi', 'dann'} or setting.startswith('adapt'):
            # We use the parameter num_train_pi_target to control
            # the amount of target data DANN gets to see
            
            csv_data_source = getattr(self, 'data_train_source')
            csv_pi_source = self._sample_pi('source')
            
            csv_data_target = getattr(self, 'data_train_target')
            csv_pi_target = self._sample_pi('target')
            csv_data_target = csv_data_target[
                csv_data_target.imageindex.isin(csv_pi_target.imageindex)
            ]

            csv_data = pd.concat(
                (csv_data_source, csv_data_target),
                ignore_index=True,
            )
            
            csv_pi = pd.concat(
                (csv_pi_source, csv_pi_target),
                ignore_index=True,
            )
        
        if setting == 'dann' or setting.startswith('adapt'):
            csv_pi = None

        return csv_data, csv_pi
