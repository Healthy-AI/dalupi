import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import join
from dalupi.models.models import create_model, create_adapt_model
from dalupi.models.utils import load_model_parameters
from dalupi.models.utils import load_adapt_model
from dalupi.data.utils import format_adapt_data
from dalupi.models.utils import evaluate
from dalupi.data.utils import create_dataset_from_config
from dalupi.data.mnist.utils import get_mnist_train_data, get_mnist_eval_data
from dalupi.data.celeb.utils import (
    get_celeb_train_data,
    get_celeb_val_data,
    get_celeb_eval_data
)

def train_model(config, setting, pipeline=None, finetune=False, **kwargs):
    experiment = config['experiment']

    fit_params = {}

    if experiment == 'chestxray' or experiment == 'coco':
        X, y = pipeline.get_data(setting=setting, **kwargs)
    elif experiment == 'mnist':
        X, y = get_mnist_train_data(config, setting=setting, **kwargs)
    elif experiment == 'celeb':
        X, y = get_celeb_train_data(config, setting=setting, **kwargs)
        X_valid, y_valid = get_celeb_val_data(config, setting)
        fit_params.update({'X_valid': X_valid, 'y_valid': y_valid})
    else:
        raise ValueError('Unknown experiment %s.' % experiment)
    
    if experiment == 'celeb' and setting == 'dalupi':
        _Xs, Xt, ws, wt = X
        _Xs_valid, Xt_valid, ws_valid, wt_valid = X_valid
        if config['x2w']['use_source_pi']:
            Xs_x2w, Xt, ws_x2w, wt = get_celeb_train_data(
                config, setting, use_source_pi=True
            )[0]
            Xt = torch.cat([Xt, Xs_x2w])
            wt = torch.cat([wt, ws_x2w])
        x2w = create_model(config, setting='x2w', finetune=finetune)
        x2w.fit(Xt, wt, X_valid=Xt_valid, y_valid=wt_valid)
        w2y = create_model(config, setting='w2y', finetune=finetune)
        w2y.fit(ws, y, X_valid=ws_valid, y_valid=y_valid)
        return [x2w, w2y]
    elif experiment == 'mnist' and setting == 'dalupi':
        Xs, Xt, ws, wt = X
        x2w = create_model(config, setting='x2w', finetune=finetune)
        x2w.fit(Xt, wt)
        w2y = create_model(config, setting='w2y', finetune=finetune)
        w2y.fit([Xs, ws], y)
        return [x2w, w2y]
    elif setting.startswith('adapt'):
        Xs_train, y_train, Xt_train, Xys_valid = format_adapt_data(X, y, config, train=True)
        if isinstance(Xys_valid, tf.data.Dataset):
            batch_size = config[setting]['model']['batch_size']
            Xys_valid = Xys_valid.batch(batch_size)
        if config[setting]['model']['random_state'] is not None:
            tf.keras.utils.set_random_seed(
                config[setting]['model']['random_state']
            )
            if config[setting]['model']['encoder']['weights'] is None:
                # FusedBatchNormGradV3 is called when pretrained weights are
                # used, and it cannot be run deterministically when training is
                # disabled (i.e., when evaluating).
                tf.config.experimental.enable_op_determinism()
        model = create_adapt_model(config, setting, finetune)
        model.compile(run_eagerly=True)
        model.fit(
            X=Xs_train,
            y=y_train,
            Xt=Xt_train,
            validation_data=Xys_valid
        )
        dataset = create_dataset_from_config(config, X, y)
        model.classes_ =  dataset.class_labels
        model.n_classes_ = len(model.classes_)
        return model
    else:
        model = create_model(config, setting, finetune)
        return model.fit(X, y, **fit_params)

def collect_scores(model, X, y, metrics, columns=[], data=[], use_adapt=False):
    scores = dict.fromkeys(metrics)

    for metric in scores.keys():
        if use_adapt:
            from_logits = model.task_.layers[-1].activation == tf.keras.activations.linear
            score = [evaluate(metric, y, model.predict(X), from_logits, average='macro')]
            class_scores = evaluate(metric, y, model.predict(X), from_logits, average=None)
        else:
            score = [model.score(X, y, metric=metric, average='macro')]
            class_scores = model.score(X, y, metric=metric, average=None)
        # We expect class_scores to be a list
        if isinstance(class_scores, float):
            class_scores = model.n_classes_ * [float('nan')]            
        score = np.append(score, class_scores)
        scores[metric] = score

    if len(class_scores) == model.n_classes_ - 1:
        # No score for background class
        classes = model.classes_[1:]
    else:
        classes = model.classes_
    classes = [c.lower().replace(' ', '_') for c in classes]
    columns += [metric + '_' + suffix for metric in scores.keys() for suffix in ['average'] + classes]
    data.extend(np.ravel(list(scores.values())))
    scores = pd.Series(data=data, index=columns)
    
    return scores

def predict_model(
    config,
    setting,
    model=None,
    pipeline=None,
    prediction_domain='target',
    subset='test',
    metrics=['auc'],
    plot_roc_curves=True
):
    use_adapt = setting.startswith('adapt')

    experiment = config['experiment']

    if use_adapt:
        # Always load best model
        model = load_adapt_model(config)
    else:
        if model is None and experiment in ['celeb', 'mnist']:
            model = [
                create_model(config, setting='x2w'),
                create_model(config, setting='w2y')
            ]
            model = [
                load_model_parameters(model[0], '_x2w'),
                load_model_parameters(model[1], '_w2y')
            ]
        elif model is None and experiment in ['chestxray', 'coco']:
            model = create_model(config, setting)
            model = load_model_parameters(model)

    if experiment == 'chestxray' or experiment == 'coco':
        X, y = pipeline.get_data(subset=subset, setting=setting, prediction_domain=prediction_domain)
    elif experiment == 'mnist':
        if setting == 'dalupi':
            get_splits = model[0].get_split_datasets
        elif use_adapt:
            get_splits = None
        else:
            get_splits = model.get_split_datasets
        X, y = get_mnist_eval_data(config, setting, subset, prediction_domain, get_splits)
        if setting == 'dalupi':
            w_hat = torch.stack([torch.from_numpy(w) for w in model[0].predict(X)])
            model = model[1]
            X = [X, w_hat]
    elif experiment == 'celeb':
        X, y = get_celeb_eval_data(config, setting, subset, prediction_domain)
        if setting == 'dalupi':
            # Since we use the CrossEntropyLoss instead of the BCEWithLogitsLoss,
            # we collect `w_hat` by calling `predict_proba` instead of `predict`.
            #
            # Each element of `w_hat` is the probability that the corresponding
            # feature is present in the image.
            w_hat = torch.stack([torch.from_numpy(w) for w in model[0].predict_proba(X)])
            model = model[1]
            X = w_hat.float()
    
    if use_adapt:
        dataset = create_dataset_from_config(config, X, y)
        model.classes_ =  dataset.class_labels
        model.n_classes_ = len(model.classes_)
        X, y = format_adapt_data(X, y, config, train=False)
        if isinstance(X, tf.data.Dataset):
            batch_size = config[setting]['model']['batch_size']
            X = X.batch(batch_size)

    columns = ['setting', 'domain', 'subset']
    data = [setting, prediction_domain, subset]
    scores = collect_scores(model, X, y, metrics, columns, data, use_adapt)

    scores_file = join(config['results']['path'], 'scores.csv')
    try:
        df = pd.read_csv(scores_file)
        df = pd.concat([df, scores.to_frame().T], ignore_index=True)
        df.to_csv(scores_file, index=False)
    except FileNotFoundError:
        df = pd.DataFrame(scores)
        df.T.to_csv(scores_file, index=False)
    
    if plot_roc_curves:
        if use_adapt:
            from_logits = model.task_.layers[-1].activation == tf.keras.activations.linear
            roc_curves = evaluate('roc', y, model.predict(X), from_logits)
        else:
            roc_curves = model.score(X, y, metric='roc')
        roc_curves.suptitle('Average AUC %.2f' % scores.auc_average)
        roc_curves.savefig(join(config['results']['path'], 'roc_curves_%s.pdf' % prediction_domain))
