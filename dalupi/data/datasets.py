import os
import torch
import torchvision
import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
from skimage.io import imread
from dalupi.models import transforms as T

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, transforms=None):
        self.csv, self.csv_pi = X
       
        if transforms is None:
            self.transforms = None
        elif isinstance(transforms, (T.Compose, torchvision.transforms.Compose)):
            self.transforms = transforms
        elif isinstance(transforms, dict):
            self.transforms_train = T.Compose(transforms['train'])
            self.transforms_eval = T.Compose(transforms['eval'])
        elif isinstance(transforms, list):
            self.transforms = torchvision.transforms.Compose(transforms)
        else:
            raise ValueError('transforms must be either a composed transforms object, a dict, a list or None.')
        
        super(BaseDataset, self).__init__()

    @property
    def class_labels(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    def decide_transform(self, training):
        if hasattr(self, 'transforms_train') and hasattr(self, 'transforms_eval'):
            self.transforms = self.transforms_train if training else self.transforms_eval
    
    def binarize_annotations(self):
        annotations = []
        default = len(self.class_labels) * [0]
        for index in self.csv.imageindex:
            if self.csv_pi is not None and index in self.csv_pi.imageindex.values:
                annotated_findings = self.csv_pi[self.csv_pi.imageindex == index]['findinglabel'].values
                annotations += [[int(p in annotated_findings) for p in self.class_labels]]
            else:
                annotations += [default]
        self.annotations = np.array(annotations)
    
    def describe(self, training=None, save_to=None, suffix=None):
        source_mask = (self.csv.domain == 'source').values
        target_mask = ~source_mask
        
        if self.csv_pi is not None:
            pi_mask = self.csv.imageindex.isin(self.csv_pi.imageindex)
        else:
            pi_mask = np.full(len(self.labels), False)
        
        if training is not None:
            assert 'valid_split' in self.csv.columns
            if training:
                if suffix is None: suffix = 'train'
                mask_train = (~self.csv.valid_split)
            else:
                if suffix is None: suffix = 'valid'
                mask_train = self.csv.valid_split
        else:
            if suffix is None: suffix = 'train'
            mask_train = np.full(len(self.labels), True)

        self.binarize_annotations()

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        x = np.arange(len(self.class_labels))
        w = 0.4
        
        for i, ax in enumerate(axes):
            if i == 0:
                # source
                mask_domain = source_mask
                mask_domain_pi = source_mask & pi_mask
                title = '# images (with PI) in source: %d (%d)'
            elif i == 1:
                # target
                mask_domain = target_mask
                mask_domain_pi = target_mask & pi_mask
                title = '# images (with PI) in target: %d (%d)'
            elif i == 2:
                # total
                mask_domain = np.full(len(self.labels), True)
                mask_domain_pi = pi_mask
                title = '# images (with PI) in total: %d (%d)'
            
            n_images = len(self.csv[mask_domain & mask_train])
            n_images_pi = len(self.csv[mask_domain_pi & mask_train])
            y1 = [self.labels[mask_domain & mask_train, i].sum() for i in x]
            y2 = [self.annotations[mask_domain & mask_train, i].sum() for i in x]

            ax.bar(x-w/2, y1, width=w, label='findings (y)')
            ax.bar(x+w/2, y2, width=w, label='annotations (w)')
            ax.set_title(title % (n_images, n_images_pi))
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_labels, rotation=45)
            ax.legend()
        
        if save_to is not None:
            fig.savefig(os.path.join(save_to, 'labels1_%s.pdf' % suffix), bbox_inches='tight')
        
        label_dist = pd.DataFrame(
            self.labels,
            index=self.csv.index,
            columns=self.class_labels
        )[mask_train].value_counts(normalize=True)
        label_dist = label_dist.reset_index()
        label_dist.rename(columns={0: 'Fraction'}, inplace=True)
        
        fig, ax = plt.subplots()
        ax.axis('off')
        table(ax, label_dist, loc='center')
        if save_to is not None:
            fig.savefig(os.path.join(save_to, 'labels2_%s.pdf' % suffix), bbox_inches='tight')

    def getitem(self, idx):
        assert isinstance(self.csv.index, pd.RangeIndex)
        imgindex = self.csv['imageindex'].iloc[idx]
        img = imread(imgindex)
        if not (len(img.shape) == 3 and img.shape[-1] == 3):
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img[:, :, np.newaxis]  # numpy.ndarray of shape (H x W x C)

        sample = {}
        sample['img'] = img
        sample['lab'] = self.labels[idx]
        sample['target'] = self.get_target(imgindex)
        sample['domain'] = self.csv['domain'].iloc[idx]
        
        if self.transforms is not None:
            try:
                sample['img'], sample['target'] = self.transforms(sample['img'], sample['target'])
            except TypeError:
                sample['img'] = self.transforms(sample['img'])

        return sample
    
    def get_target(self, imgindex):
        '''
        We know that the image has not been resized. Hence, no scaling is required.
        '''
        if self.csv_pi is None:
            return None

        # There may be more than one mask per sample
        masks = self.csv_pi[self.csv_pi['imageindex'] == imgindex]

        nonbg_no_pi = False
        
        if len(masks) == 0:
            # No PI is available for this image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            i_image = self.csv[self.csv['imageindex'] == imgindex].index.item()
            findings = self.labels[i_image]
            labels = torch.from_numpy(np.flatnonzero(findings))
            i_bg = self.class_labels.index('No Finding')
            if findings[i_bg] == 0:
                assert findings.sum() > 0
                # There are non-annotated findings
                box_loss_mask = 0
                nonbg_no_pi = True
            else:
                # Background example
                assert np.delete(findings, i_bg).sum() == 0
                box_loss_mask = 1
        else:
            boxes, labels = [], []
            for i in range(len(masks)):
                row = masks.iloc[i]
                if row['findinglabel'] in self.class_labels:
                    boxes += [[row.x, row.y, row.x+row.w, row.y+row.h]]
                    j = self.class_labels.index(row['findinglabel'])
                    labels += [j]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            box_loss_mask = 1
        
        domain = self.csv[self.csv.imageindex == imgindex]['domain']
        cls_loss_mask = int(domain == 'source') if not nonbg_no_pi else -1
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'box_loss_mask': box_loss_mask,
            'cls_loss_mask': cls_loss_mask
        }
        
        return target
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        sample = self.getitem(i)
        Xi = sample['img']
        yi = sample['lab']
        yi = torch.as_tensor(yi, dtype=torch.float32)  # Assume BCE loss
        return Xi, yi

class DALUPIDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, i):
        sample = self.getitem(i)
        Xi = sample['img']
        yi = sample['target']
        return Xi, yi

class DANNDataset(BaseDataset):
    '''
    This dataset makes several assumptions about the data csv file:
        - The csv file should contain both training and validation data during training.
            - The training data from the source domain should come first.
            - The training data from the target domain should come next.
            - The validation data should come last.
            - The validation data should come from the source domain.
        - The csv file should contain data from only one domain at test time.
    
    This dataset does only work with lupida.data.utils.TrainValidSplit.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source_mask = self.csv['domain'] == 'source'
        self.target_mask = self.csv['domain'] == 'target'
        self.n_source_samples = self.source_mask.sum()
        self.n_target_samples = self.target_mask.sum()

        subsets = self.csv.subset.unique()
        if np.array_equal(subsets, ['train', 'valid']):
            self.i_last_train = self.csv[self.csv.subset == 'train'].last_valid_index()
            self.i_first_valid = self.csv[self.csv.subset == 'valid'].first_valid_index()
            assert self.i_first_valid > self.i_last_train
            self.train = True
            self.length = self.n_source_samples
        elif np.array_equal(subsets, ['test']) or np.array_equal(subsets, ['valid']):
            self.train = False
            self.length = len(self.labels)
        else:
            raise ValueError

    def __len__(self):
        return self.length
    
    def get_train_item(self, i):
        # Collect the ith example from the source domain
        if i > self.i_last_train:
            i -= self.n_target_samples
            assert self.csv[self.source_mask].iloc[i].subset == 'valid'
        else:
            assert self.csv.iloc[i].subset == 'train'
        i = self.csv[self.source_mask].iloc[[i]].index  # isinstance(self.csv.index, pd.RangeIndex) == True
        i = list(i)[0]
        assert self.csv.iloc[i].domain == 'source'
        source_sample = self.getitem(i)
        Xs = source_sample['img']
        ys = source_sample['lab']
        ys = np.append(ys, 0)

        # Randomly sample an example from the target domain
        j = torch.randint(0, self.n_target_samples, (1,)).item()
        j = self.csv[self.target_mask].iloc[[j]].index
        j = list(j)[0]
        assert self.csv.iloc[j].domain == 'target'
        target_sample = self.getitem(j)
        Xt = target_sample['img']
        yt = target_sample['lab']
        yt = np.append(yt, 1)

        X = torch.stack([Xs, Xt])
        y = np.stack([ys, yt])

        y = torch.as_tensor(y, dtype=torch.long)

        return X, y
    
    def get_test_item(self, i):
        sample = self.getitem(i)
        Xi = sample['img']
        yi = sample['lab']
        di = 1 if sample['domain'] == 'target' else 0
        yi = np.append(yi, di)
        yi = torch.as_tensor(yi, dtype=torch.float32)  # Assume BCE loss
        return Xi, yi

    def __getitem__(self, i):
        if self.train:
            return self.get_train_item(i)
        else:
            return self.get_test_item(i)
