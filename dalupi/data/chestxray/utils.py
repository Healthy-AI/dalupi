import sklearn
import pandas as pd
import numpy as np
from dalupi.data.utils import BaseSplit

class PISplit(BaseSplit):
    '''
    Split data so that the test set only contains
    examples with privileged information. Useful for
    debugging purposes (e.g., visualizing predicted
    bounding boxes).
    '''
    def __init__(self, test_size, random_state):
        if not isinstance(test_size, int):
            raise TypeError('test_size must be an integer.')
        super().__init__(n_splits=1, train_size=test_size, random_state=random_state)
    
    def split(self, dataset):
        assert isinstance(dataset.csv.index, pd.RangeIndex)
        gsplitter = sklearn.model_selection.GroupShuffleSplit(
            n_splits=1,
            train_size=self.train_size,  # number of groups
            random_state=self.random_state
        )
        has_pi = dataset.csv.imageindex.isin(dataset.csv_pi.imageindex)
        vc = dataset.csv.patientid.value_counts()
        has_one_img = dataset.csv.patientid.isin(vc.index[vc.eq(1)])
        is_target = dataset.csv.domain == 'target'
        mask = has_pi & has_one_img & is_target  # there is actually no need to use GroupShuffleSplit
        groups = dataset.csv['patientid'][mask]
        args = (np.zeros(len(groups)), None, groups)
        ii_test, _ = next(gsplitter.split(*args))

        # We need to find the matching indexes in the full dataframe
        ii_test = dataset.csv[mask].iloc[ii_test].index
        assert all(dataset.csv.iloc[ii_test].imageindex.isin(dataset.csv_pi.imageindex))

        test_fold = np.full(len(dataset), -1)
        test_fold[ii_test] = 0

        splitter = sklearn.model_selection.PredefinedSplit(test_fold)
        yield from splitter.split()
