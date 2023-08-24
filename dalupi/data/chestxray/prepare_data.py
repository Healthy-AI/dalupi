import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from os.path import join
from sklearn.preprocessing import MultiLabelBinarizer
from .datasets import BaseDataset

C_INFO = ['imageindex', 'patientid', 'view', 'domain']
OVERLAPPING_PATHOLOGIES = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion']

NIH_CSV_FILE = 'Data_Entry_2017_v2020.csv'
NIH_PI_FILE = 'BBox_List_2017.csv'
CHEXPERT_CSV_TRAIN_FILE = 'train.csv'
CHEXPERT_CSV_VALID_FILE = 'valid.csv'
CHEXPERT_PI_FILE = 'gt_annotations_val.json'

def csv_align(nih_path, nih_domain, chexpert_path, chexpert_domain):
    nih = pd.read_csv(join(nih_path, NIH_CSV_FILE))
    
    new_nih = {}
    new_nih['imageindex'] = nih['Image Index'].apply(lambda s: join(nih_path, 'images', s))
    new_nih['patientid'] = nih['Patient ID'].astype(str)
    new_nih['view'] = nih['View Position']
    new_nih['domain'] = len(nih) * [nih_domain]
    new_nih = pd.DataFrame(new_nih)
    
    mb = MultiLabelBinarizer()
    binary_labels = mb.fit_transform(nih['Finding Labels'].apply(lambda x: x.split('|')))
    binary_labels = pd.DataFrame(binary_labels, columns=mb.classes_, index=nih.index)
    new_nih = pd.concat((new_nih, binary_labels), axis=1)
    new_nih.drop(columns=[c for c in mb.classes_ if not c in OVERLAPPING_PATHOLOGIES], inplace=True)
    new_nih.to_csv(join(nih_path, 'data.csv'), index=False)

    nih_pi = pd.read_csv(
        join(nih_path, NIH_PI_FILE),
        names=['Image Index', 'Finding Label', 'x', 'y', 'w', 'h'],
        usecols=range(6),
        skiprows=1
    )
    nih_pi.rename(columns={'Image Index': 'imageindex'}, inplace=True)
    nih_pi['imageindex'] = nih_pi['imageindex'].apply(lambda s: join(nih_path, 'images', s))
    nih_pi.loc[nih_pi['Finding Label'] == 'Infiltrate', 'Finding Label'] = 'Infiltration'
    nih_pi.rename(columns={'Finding Label': 'findinglabel'}, inplace=True)
    nih_pi = nih_pi[nih_pi.findinglabel.isin(OVERLAPPING_PATHOLOGIES)]
    nih_pi.to_csv(join(nih_path, 'pi.csv'), index=False)

    chexpert_train = pd.read_csv(join(chexpert_path, CHEXPERT_CSV_TRAIN_FILE))
    chexpert_valid = pd.read_csv(join(chexpert_path, CHEXPERT_CSV_VALID_FILE))

    for split, df in {'train': chexpert_train, 'valid': chexpert_valid}.items():
        df['view'] = df['Frontal/Lateral']
        df.loc[(df['view'] == 'Frontal'), 'view'] = df['AP/PA']
        df['view'] = df['view'].replace({'Lateral': 'L'})
        if split == 'train':
            patientid = df.Path.str.split('train/', expand=True)[1]
        elif split == 'valid':
            patientid = df.Path.str.split('valid/', expand=True)[1]
        pi_imageindex = patientid.str.replace('.jpg', '', regex=False)
        pi_imageindex = pi_imageindex.str.replace('/', '_')
        patientid = patientid.str.split('/study', expand=True)[0]
        patientid = patientid.str.replace('patient', '')
        df['patientid'] = patientid
        df['imageindex'] = df.Path.str.split('small/', expand=True)[1].apply(
            lambda s: join(chexpert_path, s)
        )
        df['pi_imageindex'] = pi_imageindex
    
    chexpert = pd.concat((chexpert_train, chexpert_valid), axis=0)
    pi_imageindex = chexpert['pi_imageindex']
    chexpert.rename(columns={'Pleural Effusion': 'Effusion'}, inplace=True)
    chexpert.drop(columns=[c for c in chexpert.columns if not c in C_INFO + OVERLAPPING_PATHOLOGIES], inplace=True)
    chexpert.insert(0, 'domain', len(chexpert) * [chexpert_domain])
    c_first = C_INFO
    c_last = sorted([c for c in chexpert.columns if not c in c_first])
    chexpert = chexpert[c_first + c_last]
    chexpert.to_csv(join(chexpert_path, 'data.csv'), index=False)

    idxmapper = dict(zip(pi_imageindex, chexpert.imageindex))
    return idxmapper

def collect_chexpert_pi(data_path, idxmapper, n_figs=0):
    with open(join(data_path, CHEXPERT_PI_FILE)) as f:
        chexpert_annotations = json.load(f)

    csv = pd.read_csv(join(data_path, 'data.csv'))
    dataset = BaseDataset((csv, None), views='*', transforms=None)

    chexpert_pi = []

    fig_counter = 0
    
    for pid, findings in chexpert_annotations.items():
        imgindex = idxmapper[pid]
        i = dataset.csv[dataset.csv.imageindex == imgindex].index.item()
        img = dataset[i][0]
        orig_img_size = findings.pop('img_size')
        hscale = img.shape[0] / orig_img_size[0]
        wscale = img.shape[1] / orig_img_size[1]
        assert math.isclose(hscale, wscale, rel_tol=0.02)
        scale = hscale
        for finding, annotations in findings.items():
            for a in annotations:
                a = np.array(a)
                xmin, ymin = a.min(axis=0)
                xmax, ymax = a.max(axis=0)
                w = xmax - xmin
                h = ymax - ymin
                xywh = np.asarray([xmin, ymin, w, h])
                xywh = xywh * scale
                chexpert_pi += [[imgindex, finding] + xywh.tolist()]

            n_annotations = len(annotations)
            if not n_annotations == 1 and fig_counter < n_figs:
                _, ax = plt.subplots()
                ax.set_title(finding)
                for i in range(n_annotations):
                    annotations[i].append(annotations[i][0])
                    xs, ys = zip(*annotations[i])
                    ax.plot(xs, ys)
                    x, y, w, h = chexpert_pi[-(n_annotations-i)][2:]
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, ec='r', fc='none')
                    ax.add_patch(rect)
                fig_counter += 1
    
    if n_figs > 0:
        plt.show()

    chexpert_pi = pd.DataFrame(chexpert_pi, columns=['imageindex', 'findinglabel', 'x', 'y', 'w', 'h'])
    chexpert_pi.loc[chexpert_pi['findinglabel'] == 'Pleural Effusion', 'findinglabel'] = 'Effusion'
    chexpert_pi = chexpert_pi[chexpert_pi.findinglabel.isin(OVERLAPPING_PATHOLOGIES)]
    chexpert_pi.to_csv(join(data_path, 'pi.csv'), index=False)

def prepare_datasets(source_path, target_path):
    source_folder = source_path.split('/')[-1].lower()
    target_folder = target_path.split('/')[-1].lower()
    if 'nih' in source_folder and 'chexpert' in target_folder:
        nih_path = source_path
        nih_domain = 'source'
        chexpert_path = target_path
        chexpert_domain = 'target'
    elif 'nih' in target_folder and 'chexpert' in source_folder: 
        nih_path = target_path
        nih_domain = 'target'
        chexpert_path = source_path
        chexpert_domain = 'source'
    else:
        raise ValueError
    idxmapper = csv_align(
        nih_path,
        nih_domain,
        chexpert_path,
        chexpert_domain
    )
    collect_chexpert_pi(chexpert_path, idxmapper)
