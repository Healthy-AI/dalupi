import dalupi.data.datasets as datasets

__ALL__ = ['BaseDataset', 'DALUPIDataset', 'DANNDataset']

C_INFO = ['imageindex', 'imageid', 'domain', 'subset', 'valid_split']

class BaseDataset(datasets.BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def class_labels(self):
        return [c for c in self.csv.columns if not c in C_INFO]

    @property
    def labels(self):
        return self.csv[self.class_labels].values

class DALUPIDataset(datasets.DALUPIDataset, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DANNDataset(datasets.DANNDataset, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
