import os
import itertools
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from functools import partial
from os.path import join
from PIL import Image

def remove_grayscale_images(data, pi):
    images_to_remove = []
    for imageindex in data.imageindex:
        image = Image.open(imageindex)
        image.load()
        image = np.asarray(image, dtype='int32')
        if not (len(image.shape) == 3 and image.shape[-1] == 3):
            images_to_remove.append(imageindex)
    data = data[~data.imageindex.isin(images_to_remove)]
    pi = pi[~pi.imageindex.isin(images_to_remove)]
    return data, pi

def remove_class_images(data, pi, cats_pred, cat_limit, n_keep, seed):
    mask = (data[cat_limit] == 1) & (data[cats_pred].sum(axis=1) == 1)
    data_keep = data[mask].sample(n_keep, random_state=seed)
    data = pd.concat((data_keep, data[~mask]))
    pi = pi[pi.imageindex.isin(data.imageindex)]
    return data, pi

def get_image_path(image_folder, image_prefix, image_id):
    file = '%s.jpg' % str(image_id).zfill(12)
    file = image_prefix + file
    return os.path.join(image_folder, file)

def get_image_ids(coco, cat_ids_domain, cat_ids_pred=None):
    image_ids = set()
    iterator = itertools.product(cat_ids_domain, cat_ids_pred) if cat_ids_pred else cat_ids_domain
    for cat_ids in iterator:
        image_ids.update(coco.getImgIds(catIds=cat_ids))
    return list(image_ids)

def get_data(coco, cats_domain, cats_pred, include_bg_samples, domain, get_image_path):
    categories = {d['id']: d['name'] for d in coco.dataset['categories']} 
    
    cat_ids_domain = coco.getCatIds(catNms=cats_domain)
    cat_ids_pred = coco.getCatIds(catNms=cats_pred)
    img_ids_fg = get_image_ids(coco, cat_ids_domain, cat_ids_pred)

    data, pi = [], []
    
    cats_pred = ['No Finding'] + cats_pred  # background class comes first
    
    if include_bg_samples > 0:
        img_ids_bg = get_image_ids(coco, cat_ids_domain)
        for img_id in img_ids_bg:
            if not img_id in img_ids_fg:
                image_path = get_image_path(img_id)
                labels = [1] + (len(cats_pred) - 1) * [0]
                data += [[image_path, domain] + labels]

    for img_id in img_ids_fg:
        labels = len(cats_pred) * [0]
        image_path = get_image_path(img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = categories[cat_id]
            if cat_name in cats_pred:
                i = cats_pred.index(cat_name)
                labels[i] = 1
                pi += [[image_path, cat_name] + ann['bbox']]
        data += [[image_path, domain] + labels]
    
    data_columns = ['imageindex', 'domain'] + cats_pred
    data = pd.DataFrame(data, columns=data_columns)
    
    pi_columns = ['imageindex', 'findinglabel', 'x', 'y', 'w', 'h']
    pi = pd.DataFrame(pi, columns=pi_columns)
    
    return data, pi

def get_matching_data(
    sup_cats_domain,
    cats_domain,
    cats_pred,
    include_bg_samples,
    dataset_path,
    version,
    domain
):
    all_data, all_pi = [], []
    
    for split in ['train', 'val']:
        split_version = split + str(version)
        
        image_folder = join(dataset_path, split_version)
        image_prefix = 'COCO_%s2014_' % split if version == 2014 else ''
        get_image_path_ = partial(get_image_path, image_folder, image_prefix)
        
        annotation_file = join(dataset_path, 'annotations', 'instances_%s.json' % split_version)
        coco = COCO(annotation_file)
        
        cats_domain_ = cats_domain + \
            [c['name'] for c in coco.dataset['categories'] if c['supercategory'] in sup_cats_domain]
        
        data, pi = get_data(coco, cats_domain_, cats_pred, include_bg_samples, domain, get_image_path_)
        
        all_data.append(data)
        all_pi.append(pi)
    
    all_data = pd.concat(all_data)
    all_pi = pd.concat(all_pi)
    
    return all_data, all_pi

def remove_overlapping_images(data_source, data_target, pi_source, pi_target):
    overlapping_indexes = set(data_source.imageindex).intersection(data_target.imageindex)
    
    data_source = data_source[~data_source.imageindex.isin(overlapping_indexes)]
    data_target = data_target[~data_target.imageindex.isin(overlapping_indexes)]
    
    pi_source = pi_source[~pi_source.imageindex.isin(overlapping_indexes)]
    pi_target = pi_target[~pi_target.imageindex.isin(overlapping_indexes)]
    
    return data_source, data_target, pi_source, pi_target

def limit_background_samples(data, n_background, seed):
    bg_mask = data['No Finding'] == 1
    data_source_bg = data[bg_mask]
    data_source_fg = data[~bg_mask]
    n_background = min([n_background, len(data_source_bg)])
    data_source_bg = data_source_bg.sample(n_background, random_state=seed)
    return pd.concat([data_source_bg, data_source_fg])

def prepare_datasets(
    super_cats_source,
    cats_source,
    super_cats_target,
    cats_target,
    cats_pred,
    n_background,
    dataset_path,
    version=2017,
    seed=0
):
    include_bg_samples = n_background > 0
    
    data_source, pi_source = get_matching_data(
        super_cats_source,
        cats_source,
        cats_pred,
        include_bg_samples,
        dataset_path,
        version,
        domain='source'
    )

    data_target, pi_target = get_matching_data(
        super_cats_target,
        cats_target,
        cats_pred,
        include_bg_samples,
        dataset_path,
        version,
        domain='target'
    )

    data_source, data_target, pi_source, pi_target = \
        remove_overlapping_images(data_source, data_target, pi_source, pi_target)
    
    if include_bg_samples:
        data_source = limit_background_samples(data_source, n_background, seed)
        data_target = limit_background_samples(data_target, n_background, seed)

    return data_source, data_target, pi_source, pi_target
