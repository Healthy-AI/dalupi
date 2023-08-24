import numpy as np
import dalupi.data.datasets as datasets

__ALL__ = ['BaseDataset', 'DALUPIDataset', 'DANNDataset']

OVERLAPPING_PATHOLOGIES = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion']

class BaseDataset(datasets.BaseDataset):
    def __init__(
        self,
        *args,
        views=['AP', 'PA'],
        uncertainty='ignore',
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.pathologies = OVERLAPPING_PATHOLOGIES
        self.views = views
        self.uncertainty = uncertainty

        # Limit data
        self.limit_to_selected_views()
        self.csv = self.csv.reset_index(drop=True)
        
        # Collect labels
        self.labels_ = [self.extract_labels(p) for p in self.pathologies]
        self.labels_ = np.asarray(self.labels_, dtype=np.float32).T

        # Deal with uncertainty
        if self.uncertainty is not None:
            mask = self.labels_ == -1
            if self.uncertainty == 'ignore':
                self.labels_[mask] = np.nan
            elif self.uncertainty == 'negative':
                self.labels_[mask] = 0
            elif self.uncertainty == 'positive':
                self.labels_[mask] = 1
            # Missing labels are treated as negative (https://github.com/stanfordmlgroup/chexpert-labeler/issues/9)
            self.labels_[np.isnan(self.labels_)] = 0
    
    @property
    def class_labels(self):
        return self.pathologies
    
    @property
    def labels(self):
        return self.labels_
    
    def limit_to_selected_views(self):
        if type(self.views) is not list:
            self.views = [self.views]
        if not '*' in self.views:
            self.csv = self.csv[self.csv['view'].isin(self.views)]
            if self.csv_pi is not None:
                self.csv_pi = self.csv_pi[self.csv_pi.imageindex.isin(self.csv.imageindex)]
    
    def extract_labels(self, pathology):
        labels = self.csv[pathology].copy()
        if pathology != 'Support Devices' and pathology != 'No Finding':
            is_healthy = self.csv['No Finding'] == 1
            labels[is_healthy] = 0
        return labels
    
    def remove(self, imageindexes, remove_patients=True):
        if remove_patients:
            pids = self.csv[self.csv.imageindex.isin(imageindexes)].patientid
            self.csv = self.csv[~self.csv.patientid.isin(pids)]
        else:
            self.csv = self.csv[~self.csv.imageindex.isin(imageindexes)]
        self.labels_ = self.labels_[self.csv.index]
        self.csv = self.csv.reset_index(drop=True)

class DALUPIDataset(datasets.DALUPIDataset, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DANNDataset(datasets.DANNDataset, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
