import os
import skorch
import torch
import warnings
import copy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import skorch.callbacks as cbs
from functools import partial
from pathlib import Path
from sklearn import metrics
from skorch.dataset import unpack_data, get_len
from amhelpers.amhelpers import create_object_from_dict
from amhelpers.config_parsing import get_net_params, _check_value
from dalupi.models.utils import save_training_progress_fig
from dalupi.models.utils import save_batch_stats_fig
from dalupi.models.utils import save_lupi_loss_terms_fig
from dalupi.models.utils import save_dann_loss_terms_fig
from dalupi.models.utils import call_dataset
from skorch.utils import to_tensor, to_numpy
from .utils import create_rectangle

def create_adapt_model(config, setting, finetune=False):
    if finetune:
        warnings.warn('Finetuning is not yet implemented for adapt models.')
    config = copy.deepcopy(config)
    model_dict = config[setting]['model']
    model_dict.pop('is_called', None)
    if 'callbacks' in model_dict:
        if isinstance(model_dict['callbacks'], dict):
            callbacks_dicts = model_dict['callbacks'].values()
        else:
            callbacks_dicts = model_dict['callbacks']
        for callback_dict in callbacks_dicts:
            if callback_dict['type'].endswith('ModelCheckpoint'):
                checkpoint_dir = os.path.join(
                    config['results']['path'],
                    'checkpoints'
                )
                Path(checkpoint_dir).mkdir(exist_ok=True)
                callback_dict['filepath'] = checkpoint_dir
    model_params = {k: _check_value(v) if not k == 'type' else v for k, v in model_dict.items()}
    model = create_object_from_dict(model_params)
    model.results_path = config['results']['path']
    return model

def create_model(config, setting, finetune=False):
    config = copy.deepcopy(config)
    
    model_dict = config[setting]['model']
    model_dict.pop('is_called', None)
    model_params = {k: _check_value(v) if not k == 'type' else v for k, v in model_dict.items()}
    net_params = get_net_params(config['default'], config[setting])
    model_params.update(net_params)
    
    model_params['results_path'] = config['results']['path']
    
    if finetune:
        lr = config['finetuning']['lr']
        max_epochs = config['finetuning']['max_epochs']
        n_layers = config['finetuning']['n_layers']

        model_params['finetune'] = True
        
        model_params['lr'] = lr
        for k in model_params.keys():
            if k.endswith('__lr'):
                model_params[k] = lr
        
        model_params['max_epochs'] = max_epochs

        if model_params['type'].endswith('DALUPI'):
            model_params['module__backbone_n_trainable_layers'] = n_layers
        else:
            model_params['module__featurizer__n_trainable_layers'] = n_layers

    return create_object_from_dict(model_params)

def custom_monitor(net):
    monitor = net.monitor.replace('_best', '')
    scores = [net.history[i][monitor] for i in range(len(net.history))]
    current_score = scores.pop(-1)
    if not scores:
        # First model is always best
        return True
    if net.lower_is_better:
        return current_score < min(scores)
    else:
        return current_score > max(scores) 

class BaseModel(skorch.net.NeuralNet):
    def __init__(
        self,
        multilabel=True,
        results_path=None,
        epoch_scoring=None,
        monitor='valid_loss_best',
        f_params='params.pt',
        f_optimizer='optimizer.pt',
        f_criterion='criterion.pt',
        f_history='history.json',
        f_pickle=None,
        load_best=True,
        patience=5,
        finetune=False,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if results_path is None:
            raise ValueError('results_path cannot be None.')
        
        if 'auc' in monitor or 'accuracy' in monitor:
            assert epoch_scoring in ['auc', 'accuracy']
            self.lower_is_better = False
        elif 'loss' in monitor:
            self.lower_is_better = True
        else:
            raise ValueError("monitor should contain either 'auc', 'accuracy' or 'loss'.")

        self.multilabel = multilabel
        self.results_path = results_path
        self.epoch_scoring = epoch_scoring
        self.monitor = monitor
        self.finetune = finetune
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if not isinstance(self, DALUPI):
                # DALUPI cannot be fully deterministic
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = False
                    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                torch.use_deterministic_algorithms(True)
        
        callbacks = []

        if hasattr(self.module, '_freeze'):
            freezer = cbs.Freezer(partial(self.module._freeze, self))
            callbacks += [freezer]         

        if epoch_scoring is not None:
            assert epoch_scoring in ['auc', 'accuracy']
            scoring = self.compute_auc if epoch_scoring == 'auc' else self.compute_accuracy
            scoring_kwargs = {
                'scoring': scoring,
                'lower_is_better': False,
                'target_extractor': self._extract_target
            }
            if isinstance(self, DANN):
                scoring_kwargs['use_caching'] = False  # Cached predictions contain null values
            train_name = 'train_%s' % epoch_scoring
            valid_name = 'valid_%s' % epoch_scoring
            callbacks += [
                (train_name, cbs.EpochScoring(name=train_name, on_train=True, **scoring_kwargs)),
                (valid_name, cbs.EpochScoring(name=valid_name, **scoring_kwargs)),
            ]

        checkpoint = cbs.Checkpoint(
            monitor=custom_monitor,
            f_params=f_params,
            f_optimizer=f_optimizer,
            f_criterion=f_criterion,
            f_history=f_history,
            f_pickle=f_pickle,
            fn_prefix='best_',
            dirname=results_path,
            load_best=load_best
        )
        callbacks += [checkpoint]
        
        early_stopping = cbs.EarlyStopping(
            monitor=monitor.replace('_best', ''),
            patience=patience,
            lower_is_better=self.lower_is_better,
            threshold=1e-3
        )
        callbacks += [early_stopping]

        if finetune:
            load_state = cbs.LoadInitState(checkpoint)
            callbacks += [load_state]
        
        if self.callbacks is None:
            self.callbacks = callbacks
        else:
            self.callbacks.extend(callbacks)

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', cbs.EpochTimer()),
            ('train_loss', cbs.PassthroughScoring(name='train_loss', on_train=True)),
            ('valid_loss', cbs.PassthroughScoring(name='valid_loss')),
            ('print_log', cbs.PrintLog('classes_'))
        ]

    def get_split_datasets(self, X, y, X_valid=None, y_valid=None, **fit_params):
        if X_valid is not None:
            dataset_train = self.get_dataset(X, y)
            dataset_valid = self.get_dataset(X_valid, y_valid)
            return dataset_train, dataset_valid
        return super().get_split_datasets(X, y, **fit_params)
    
    # Catch `fit_params` here and do not pass it on to the step function.
    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        if dataset is None:
            return

        batch_count = 0
        for batch in self.get_iterator(dataset, training=training):
            self.notify('on_batch_begin', batch=batch, training=training)
            step = step_fn(batch)
            self.history.record_batch(prefix + '_loss', step['loss'].item())
            batch_size = (get_len(batch[0]) if isinstance(batch, (tuple, list))
                          else get_len(batch))
            self.history.record_batch(prefix + '_batch_size', batch_size)
            self.notify('on_batch_end', batch=batch, training=training, **step)
            batch_count += 1
        
        self.history.record(prefix + '_batch_count', batch_count)

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self.history.new_epoch()
        self.history.record('epoch', len(self.history))
        if len(self.history) == 1:
            try:
                call_dataset(dataset_train, 'describe', training=True, save_to=self.results_path)
                call_dataset(dataset_train, 'describe', training=False, save_to=self.results_path)
            except AssertionError:
                call_dataset(dataset_train, 'describe', save_to=self.results_path)
            self.history.record('classes_', self.classes_)
        if self.finetune:
            self.history.record('finetune', True)
            if self.first_epoch:
                for name in self._optimizers:
                    optimizer = getattr(self, name + '_')
                    for g in optimizer.param_groups:
                        g['lr'] = self.lr
                self.first_epoch = False
    
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        dataset = self.get_dataset(X)
        self.classes_ = dataset.class_labels
        self.n_classes_ = len(self.classes_)
        self.first_epoch = True

    def on_train_end(self, net, X=None, y=None, **kwargs):
        save_training_progress_fig(net.history, self.results_path, metric='loss')
        if self.epoch_scoring is not None:
            save_training_progress_fig(net.history, self.results_path, metric=self.epoch_scoring)

    def get_iterator(self, dataset, training=False):
        call_dataset(dataset, 'decide_transform', training)
        return super().get_iterator(dataset, training)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        unreduced_loss = super().get_loss(y_pred, y_true, X, training)
        return torch.nan_to_num(unreduced_loss, nan=0).mean()
    
    def predict(self, X, th=0.5):
        if self.multilabel:
            probas = self.predict_proba(X)
            return (probas > th).astype(int)
        else:
            if isinstance(self.criterion_, torch.nn.MSELoss):
                return self.predict_proba(X)
            else:
                return self.predict_proba(X).argmax(axis=1)

    def compute_accuracy(self, net, X, y, **kwargs):
        return net.score(X, y, metric='accuracy', **kwargs)

    def compute_auc(self, net, X, y, **kwargs):
        if not self.multilabel and not 'multi_class' in kwargs:
            kwargs['multi_class'] = 'ovo'
        return net.score(X, y, metric='auc', **kwargs)
    
    def _extract_target(self, y):
        return to_numpy(y)
    
    def _collect_labels_from_dataset(self, dataset):
        y = []
        for _, yi in dataset:
            y.append(self._extract_target(yi))
        return np.array(y)

    def score(self, X, y=None, metric='auc', threshold=0.5, **kwargs):
        valid_metrics = {'auc', 'roc', 'hamming', 'f1', 'accuracy'}
        if not metric in valid_metrics:
            raise ValueError(
                'The specified metric is not yet accepted.'
            )

        if metric in {'auc', 'roc'}:
            yp = self.predict_proba(X)
        else:
            yp = self.predict(X, threshold)

        if y is None:
            dataset = self.get_dataset(X) if not isinstance(X, torch.utils.data.Dataset) else X
            dataset.transforms = None  # No transforms need to be applied since we only collect labels
            y = self._collect_labels_from_dataset(dataset)
        
        start = 0

        if self.multilabel:
            y = np.reshape(y, yp.shape)
            if y[:, 0].sum() == 0:
                # No background samples are available so AUC will be undefined
                y = y[:, 1:]
                yp = yp[:, 1:]
                start = 1
        else:
            y = y.reshape(-1)

        if metric == 'auc':
            try:
                return metrics.roc_auc_score(y, yp, **kwargs)
            except ValueError:
                return float('nan')
        if metric == 'roc':
            fig, axes = plt.subplots(1, y.shape[1], figsize=(20, 4))
            for i, (y_, yp_, ax) in enumerate(zip(y.T, yp.T, axes), start=start):
                fpr, tpr, _ = metrics.roc_curve(y_, yp_)
                roc_auc = metrics.auc(fpr, tpr)
                display = metrics.RocCurveDisplay(
                    fpr=fpr,
                    tpr=tpr,
                    roc_auc=roc_auc,
                    estimator_name=self.classes_[i]
                )
                display.plot(ax=ax)
            return fig
        if metric == 'f1':
            return metrics.f1_score(y, yp, **kwargs)
        if metric == 'accuracy':
            if 'average' in kwargs and kwargs['average'] is None:
                return float('nan')
            else:
                return metrics.accuracy_score(y, yp)
        if metric == 'hamming':
            return metrics.hamming_loss(y, yp)

class DALUPI(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fig_counter = 0

    @property
    def _default_callbacks(self):
        keys_ignored = [
            'classes_',
            'n_batch_proposals',
            'n_batch_positives',
            'n_batch_negatives',
            'bce_loss_classifier',
            'ce_loss_classifier',
            'loss_box_reg',
            'loss_objectness',
            'loss_rpn_box_reg'
        ]
        return [
            ('epoch_timer', cbs.EpochTimer()),
            ('train_loss', cbs.PassthroughScoring(name='train_loss', on_train=True)),
            ('valid_loss', cbs.PassthroughScoring(name='valid_loss')),
            ('print_log', cbs.PrintLog(keys_ignored))
        ]

    def _visualize_predicted_boxes(self, output, axes, max_n_boxes=5):
        boxes = output['boxes'].cpu().numpy()
        boxes = boxes.astype(int)
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        min_top_score_per_label = {}
        for label in set(labels):
            scores_label = scores[labels == label]
            if len(scores_label) > 0:
                sorted_scores_label = scores_label[np.argsort(scores_label)]
                min_top_score_per_label[label] = sorted_scores_label[-(max_n_boxes+1):][0]
            else:
                min_top_score_per_label[label] = -np.inf
        colors = sns.color_palette('tab10', max_n_boxes)
        label_counter = {k: 0 for k in set(labels)}
        for box, label, score in zip(boxes, labels, scores):
            if score > min_top_score_per_label[label]:
                i = label_counter[label]
                ax = axes[label-1]
                rectangle = create_rectangle(box, ec=colors[i])
                ax.add_patch(rectangle)
                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width() / 2.0
                cy = ry + rectangle.get_height() / 2.0
                s = round(score, 2)
                ax.annotate(s, (cx, cy), color=colors[i], ha='center', va='center')
                label_counter[label] += 1

    def save_images(self, outputs, Xi, yi):        
        str_labels = self.classes_[1:]  # ignore background label

        figdir = os.path.join(self.results_path, 'images')
        Path(figdir).mkdir(exist_ok=True)

        for i, output in enumerate(outputs):
            gt_boxes = yi[i]['boxes']
            if gt_boxes.numel() > 0:
                image = Xi[i][0].cpu().numpy()
                gt_labels = yi[i]['labels'].cpu().numpy()

                fig, axes = plt.subplots(3, len(str_labels), figsize=(20, 10))
                axes = axes.reshape(-1, axes.shape[-1])
                
                for label, str_label in enumerate(str_labels, start=1):  # start from 1 since background is 0
                    i_label = label - 1
                    axes[0, i_label].imshow(image, cmap='Greys_r')
                    axes[0, i_label].set_title(str_label)
                    
                    axes[1, i_label].imshow(image, cmap='Greys_r')
                    indexes = np.flatnonzero(gt_labels == label)
                    if len(indexes) > 0:
                        for j in indexes:
                            rectangle = create_rectangle(gt_boxes[j])
                            axes[1, i_label].add_patch(rectangle)
                    else:
                        axes[1, i_label].axis('off')

                    axes[2, i_label].imshow(image, cmap='Greys_r')
                
                self._visualize_predicted_boxes(output, axes[2])
                
                epoch = self.history[-1]['epoch']
                fname = os.path.join(figdir, 'img_e%s_f%02d' % (epoch, self.fig_counter+1))
                fig.savefig(fname, bbox_inches='tight')
                plt.close(fig)
                self.fig_counter += 1
    
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        model = net.module_.model
        
        for i, batch_dict in enumerate(net.history[-1]['batches']):
            if next(iter(batch_dict)).startswith('valid'):
                break
            batch_dict.update({'train_n_batch_proposals': model.rpn.n_batch_proposals[i]})
            batch_dict.update({'train_n_batch_positives': model.roi_heads.n_batch_positives[i]})
            batch_dict.update({'train_n_batch_negatives': model.roi_heads.n_batch_negatives[i]})
        
        model.rpn.n_batch_proposals = []
        model.roi_heads.n_batch_positives = []
        model.roi_heads.n_batch_negatives = []

        if len(self.history) % 10 == 0:
            X, y = [], []
            for batch in self.get_iterator(dataset_valid, training=False):
                inputs, targets = unpack_data(batch)
                for i, t in zip(inputs, targets):
                    if t['boxes'].numel() > 0 and len(X) < 10:
                        X.append(i)
                        y.append(t)
            if len(X) > 0:
                _, outputs = self.evaluation_step((X, y), training=False, return_outputs=True)
                self.save_images(outputs, X, y)
                self.fig_counter = 0
    
    def on_train_end(self, net, X=None, y=None, **kwargs):
        super().on_train_end(net, X=None, y=None, **kwargs)

        save_batch_stats_fig(net.history, 'train_n_batch_proposals', self.results_path, 'proposals')
        
        keys = ['train_n_batch_positives', 'train_n_batch_negatives']
        save_batch_stats_fig(net.history, keys, self.results_path, 'pos_neg')
        
        iterable = itertools.product(['train', 'valid'], self.classes_)
        keys = [a + '_max_score_' + b.lower().replace(' ', '_') for a, b in iterable]
        save_batch_stats_fig(net.history, keys, self.results_path, 'max_scores')
        
        save_lupi_loss_terms_fig(net.history, self.results_path)
    
    def _pi_step(self, batch, training, **fit_params):
        y_pred = self.evaluation_step(batch, training=False)
        prefix = 'train_' if training else 'valid_'
        for score, label in zip(torch.max(y_pred, 0)[0], self.classes_):
            label = label.lower().replace(' ', '_')
            self.history.record_batch(prefix + 'max_score_' + label, score.item())
        step = {'y_pred': y_pred}

        # Compute loss
        Xi, yi = unpack_data(batch)
        if training:
            # The above call puts the module into evaluation mode.
            # We switch back to training mode.
            self._set_training(True)
            Xi = {'images': Xi, 'targets': yi}
            loss_dict = self.infer(Xi, **fit_params)
            for k, v in loss_dict.items():
                self.history.record_batch(prefix + k, v.item())
            loss_vector = torch.stack([l for l in loss_dict.values()])
            step['loss'] = torch.nansum(loss_vector)
        else:
            # The faster R-CNN loss may not be complete due to lack 
            # of boxes in the validation data. Instead, we compute
            # the classification loss explicitly here.
            y_true = torch.Tensor(self._extract_target(yi))
            step['loss'] = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
        
        return step
    
    def train_step_single(self, batch, **fit_params):
        step = self._pi_step(batch, training=True, **fit_params)
        step['loss'].backward()
        return step
    
    def validation_step(self, batch, **fit_params):
        return self._pi_step(batch, training=False, **fit_params)

    def _compute_probas(self, outputs):
        '''
        Compute probabilities from prediction matrix.
        Source: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730052.pdf
        '''
        yps = []
        for output in outputs:  # loop over images in batch
            labels = output['labels']
            scores = output['scores']
            yp = []
            for i in range(self.n_classes_):
                ii = torch.where(labels == i)[0]
                yp += [torch.max(scores[ii]).cpu().item()] if len(ii) > 0 else [0]
            yps.append(yp)
        return torch.Tensor(yps)

    def evaluation_step(self, batch, training=False, return_outputs=False):
        self.check_is_fitted()
        self._set_training(training)
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training):
            Xi = {'images': Xi}
            outputs = self.infer(Xi)
            yp = self._compute_probas(outputs)
            if return_outputs:
                return yp, outputs
            else:
                return yp
    
    def _extract_target(self, y):
        if isinstance(y, list):
            return [self._extract_target(yi) for yi in y]
        else:
            labels = self.n_classes_ * [0]
            for l in y['labels']:
                labels[l] = 1
            return labels

def _prepare_dann_data(batch, training):
    X, y = unpack_data(batch)
    if training:
        if X.ndim > 4:
            c, w, h = X.shape[-3:]
            X = torch.reshape(X, (-1, c, w, h))
        if y.ndim > 2:
            n = y.shape[-1]
            y = torch.reshape(y, (-1, n))
    else:
        if X.ndim > 4:
            X = X[:, 0]
        if y.ndim > 2:
            y = y[:, 0]
    return X, y

class DANN(BaseModel):
    def __init__(
        self,
        optimizer_generator,
        optimizer_discriminator,
        d_steps_per_g_step=1,
        grad_penalty=0,
        theta=1,
        **kwargs,
    ):
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.d_steps_per_g_step = d_steps_per_g_step
        self.grad_penalty = grad_penalty
        self.theta = theta

        super().__init__(**kwargs)
    
    @property
    def _default_callbacks(self):
        keys_ignored = [
            'classes_',
            'train_classifier_loss',
            'valid_classifier_loss',
            'train_disc_loss',
            'valid_disc_loss'
        ]
        return [
            ('epoch_timer', cbs.EpochTimer()),
            ('train_loss', cbs.PassthroughScoring(name='train_loss', on_train=True)),
            ('valid_loss', cbs.PassthroughScoring(name='valid_loss')),
            ('print_log', cbs.PrintLog(keys_ignored))
        ]
    
    def on_train_end(self, net, X=None, y=None, **kwargs):
        super().on_train_end(net, X=None, y=None, **kwargs)
        save_dann_loss_terms_fig(net.history, self.results_path)
    
    def initialize_optimizer(self):
        generator_named_parameters = list(self.module_.featurizer.named_parameters())
        generator_named_parameters += list(self.module_.classifier.named_parameters())
        pgroups, kwargs = self.get_params_for_optimizer(
            'optimizer_generator', generator_named_parameters
        )
        self.optimizer_generator_ = self.optimizer_generator(*pgroups, **kwargs)
        
        discriminator_named_parameters = list(self.module_.discriminator.named_parameters())
        discriminator_named_parameters += list(self.module_.class_embeddings.named_parameters())
        pgroups, kwargs = self.get_params_for_optimizer(
            'optimizer_discriminator', discriminator_named_parameters
        )
        self.optimizer_discriminator_ = self.optimizer_discriminator(*pgroups, **kwargs)
        
        return self
    
    def get_loss(self, y_pred, y_true, X=None, training=False):
        if isinstance(self.criterion_, torch.nn.CrossEntropyLoss):
            if y_pred.ndim == 2:
                y_true = torch.reshape(y_true, (y_pred.shape[0],))
        elif isinstance(self.criterion_, torch.nn.BCEWithLogitsLoss):
            y_true = y_true.type(torch.float32)
        y_true = to_tensor(y_true, device=self.device)
        return self.criterion_(y_pred, y_true)
    
    def infer(self, x, y, **fit_params):
        x = to_tensor(x, device=self.device)
        y = to_tensor(y, device=self.device)
        return self.module_(x, y, **fit_params)
    
    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        Xi, yi = _prepare_dann_data(batch, training=False)
        with torch.no_grad():
            losses, y_pred = self.infer(Xi, yi, infer_classifier=True, classifier_loss_fcn=self.get_loss)
            loss = self.aggregate_lossses(losses, prefix='valid')
        return {'loss': loss, 'y_pred': y_pred}

    def aggregate_lossses(self, losses, prefix):
        disc_loss = losses['disc_loss']
        self.history.record_batch(prefix + '_disc_loss', disc_loss.item())
        if 'grad_penalty' in losses:
            disc_loss += self.grad_penalty * losses['grad_penalty']
        if 'classifier_loss' in losses:
            gen_loss = losses['classifier_loss'] + self.theta * (-disc_loss)
            self.history.record_batch(prefix + '_classifier_loss', losses['classifier_loss'].item())
            return gen_loss
        else:
            return disc_loss

    def train_step(self, batch, **fit_params):
        batch_index = len(self.history[-1]['batches'])
        update_discriminator = ((batch_index - 1) % (1 + self.d_steps_per_g_step) < self.d_steps_per_g_step)
        
        if update_discriminator:
            self.optimizer_discriminator_.zero_grad()
        else:
            self.optimizer_discriminator_.zero_grad()
            self.optimizer_generator_.zero_grad()

        self._set_training(True)
        Xi, yi = _prepare_dann_data(batch, training=True)
        losses, y_pred = self.infer(Xi, yi, infer_classifier=(not update_discriminator), classifier_loss_fcn=self.get_loss)
        loss = self.aggregate_lossses(losses, prefix='train')

        if update_discriminator:
            loss.backward()
            self.optimizer_discriminator_.step()
        else:
            loss.backward()
            self.optimizer_generator_.step()

        return {'loss': loss, 'y_pred': y_pred}
    
    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        Xi, _ = _prepare_dann_data(batch, training)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            Xi = to_tensor(Xi, device=self.device)
            return self.module_.predict(Xi)

    def _extract_target(self, y):
        y = to_numpy(y)
        if y.ndim == 1:
            return y[:-1]
        elif y.ndim == 2:
            return y[0, :-1]
        elif y.ndim == 3:
            y = y.reshape(-1, y.shape[-1])
            y = y[y[:, -1] == 0]
            return y[:, :-1]
        else:
            raise ValueError
