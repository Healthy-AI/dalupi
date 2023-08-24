import os
import pickle
import torch
import sys
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.metrics as metrics
from functools import partial
from os.path import join
from matplotlib.ticker import MaxNLocator
from skorch.callbacks import Callback
from torch.optim.lr_scheduler import StepLR, LambdaLR
from scipy.special import softmax, expit
from sklearn import preprocessing
from skorch.dataset import unpack_data

class LoadBestModel(tensorflow.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        # Assuming save_best_only=True
        self.model = tensorflow.keras.models.load_model(
            os.path.join(
                self.model.results_path,
                'checkpoints'
            )
        )

class SaveTrainingProgress(tensorflow.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        metrics = [k for k in logs.keys() if not k.startswith('val_')]
        for metric in metrics:
            fig, ax = plt.subplots()
            ax.plot(self.model.history.history[metric], label=metric)
            val_metric = 'val_' + metric
            if val_metric in logs.keys():
                ax.plot(self.model.history.history[val_metric], label=val_metric)
            ax.legend()
            ax.set_xlabel('Epoch')
            file = join(self.model.results_path, '%s_progress.pdf' % metric)
            _save_fig(fig, file)

def evaluate(metric, y_true, y_pred, from_logits, **kwargs):
    '''
    Evaluate ADAPT models.
    '''
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()  # logits
    
    y_true = y_true.reshape(y_pred.shape[0], -1)
    
    if from_logits:
        if y_true.shape[1] > 1:
            # Multi-label classification
            if y_true[:, 0].sum() == 0:
                # No background samples are available so AUC will be undefined
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]
            y_pred = expit(y_pred)
        elif y_pred.shape[1] > 1:
            # Multi-class classification
            y_pred = np.apply_along_axis(softmax, axis=1, arr=y_pred)
        else:
            raise ValueError('Unknown problem type.')
    
    if metric == 'auc':
        try:
            return metrics.roc_auc_score(y_true, y_pred, **kwargs)
        except ValueError as e:
            return float('nan')
    elif metric == 'roc':
        fig, axes = plt.subplots(1, y_true.shape[1], figsize=(20, 4))
        for y, yp, ax in zip(y_true.T, y_pred.T, axes):
            fpr, tpr, _ = metrics.roc_curve(y, yp)
            roc_auc = metrics.auc(fpr, tpr)
            display = metrics.RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc
            )
            display.plot(ax=ax)
        return fig
    elif metric == 'accuracy':
        y_pred = np.argmax(y_pred, axis=1)
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            lb = preprocessing.LabelBinarizer()
            lb.fit(range(y_true.shape[1]))
            y_pred = lb.transform(y_pred)
        if 'average' in kwargs and kwargs['average'] is None:
            return float('nan')
        else:
            return metrics.accuracy_score(y_true, y_pred)
    else:
        raise ValueError('The specified metric is not yet accepted.')

class Scorer:
    def __init__(self, metric, name=None, **kwargs):
        self.__name__ = name if name is not None else metric
        self.scorer = partial(evaluate, metric, **kwargs)

    def __call__(self, y_true, y_pred):
        return self.scorer(y_true, y_pred)

def make_accept_grayscale(first_layer, requires_grad=False):
    '''https://datascience.stackexchange.com/questions/65783/pytorch-how-to-use-pytorch-pretrained-for-single-channel-image'''    
    w = first_layer.weight
    assert not first_layer.bias
    new_first_layer = torch.nn.Conv2d(
        in_channels=1,
        out_channels=first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=False
    )
    new_first_layer.weight = torch.nn.Parameter(torch.mean(w, dim=1, keepdim=True), requires_grad=requires_grad)
    return new_first_layer

def call_dataset(dataset, method_name, *args, **kwargs):
    '''
    Call a method of a dataset which may be wrapped by torch.utils.data.Subset.
    In such a case, the dataset can be reached using dataset.dataset.
    '''
    if hasattr(dataset, method_name):
        return getattr(dataset, method_name)(*args, **kwargs)
    else:
        return getattr(dataset.dataset, method_name)(*args, **kwargs)

def load_model_parameters(
    model,
    suffix=''
):
    try:
        f_best_hyperparams = join(model.results_path, 'best_hyperparams.pickle')
        with open(f_best_hyperparams, 'rb') as f:
            best_hyperparams = pickle.load(f)
        model.set_params(**best_hyperparams)
    except FileNotFoundError:
        pass
    
    if not model.initialized_:
        model.initialize()

    f_params = join(model.results_path, 'best_params%s.pt' % suffix)
    f_optimizer = join(model.results_path, 'best_optimizer%s.pt' % suffix)
    f_criterion = join(model.results_path, 'best_criterion%s.pt' % suffix)
    f_history = join(model.results_path, 'best_history%s.json' % suffix)

    if not os.path.isfile(f_params): f_params = None
    if not os.path.isfile(f_optimizer): f_optimizer = None
    if not os.path.isfile(f_criterion): f_criterion = None
    if not os.path.isfile(f_history): f_history = None

    model.load_params(f_params, f_optimizer, f_criterion, f_history)
    
    if 'classes_' in model.history[0]:
        model.classes_ = model.history[0]['classes_']
        model.n_classes_ = len(model.classes_)

    return model

def load_adapt_model(config):
    try:
        model = tensorflow.keras.models.load_model(
            join(config['results']['path'], 'checkpoints')
        )
    except ValueError:
        from dalupi.models.utils import MyDecay
        model = tensorflow.keras.models.load_model(
            join(config['results']['path'], 'checkpoints'),
            custom_objects={'MyDecay': MyDecay}
        )
    return model

def auc_scorer(model, X, y=None):
    return model.score(X, y, metric='auc', average='macro')

def f1_scorer(model, X, y=None):
    return model.score(X, y, metric='f1', average='macro')

def plot_stats(history, ykey, from_batches, ax, **kwargs):
    x, y = [], []
    epoch_start_finetune = None
    for i, h in enumerate(history, start=1):
        finetune = h['finetune'] if 'finetune' in h else None
        if finetune and not epoch_start_finetune:
            epoch_start_finetune = i
        if from_batches:
            for hi in h['batches']:    
                try:
                    y.append(hi[ykey])  # this should come first, otherwise x may be appended even if y is not
                    x.append(h['epoch'])
                except KeyError:
                    pass
        else:
            try:
                x.append(h['epoch'])
                y.append(h[ykey])
            except KeyError:
                pass
    if len(y) > 0 and not (all(np.isnan(y)) or all(np.isinf(y))):
        sns.lineplot(x=x, y=y, ax=ax, **kwargs)
        if epoch_start_finetune is not None:
            ax.axvline(x=epoch_start_finetune, c='black', ls='--', lw=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def _save_fig(fig, file):
    if os.path.exists(file):
        split = file.split('.')
        extension = split[-1]
        file = ''.join(split[:-1])
        file += '_2.' + extension
    fig.savefig(file, bbox_inches='tight')
    plt.close(fig)

def save_training_progress_fig(history, path, metric='loss'):
    fig, ax = plt.subplots()
    plot_stats(history, 'train_' + metric, False, ax, label='train')
    plot_stats(history, 'valid_' + metric, False, ax, label='valid')
    ax.legend()
    ax.set_ylabel(metric)
    file = join(path, metric + '_progress.pdf')
    _save_fig(fig, file)

def save_dann_loss_terms_fig(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = axes.flatten()
    for i, loss_term in enumerate(
        [
            'disc_loss',
            'classifier_loss'
        ]
    ):
        plot_stats(history, 'train_' + loss_term, True, axes[i], label='train')
        plot_stats(history, 'valid_' + loss_term, True, axes[i], label='valid')
        axes[i].set_title(loss_term)
    file = join(path, 'loss_terms.pdf')
    _save_fig(fig, file)

def save_lupi_loss_terms_fig(history, path):
    fig, axes = plt.subplots(3, 2, figsize=(20, 10))
    axes = axes.flatten()
    for i, loss_term in enumerate(
        [
            'bce_loss_classifier',
            'ce_loss_classifier',
            'loss_box_reg',
            'loss_objectness',
            'loss_rpn_box_reg',
        ]
    ):
        plot_stats(history, 'train_' + loss_term, True, axes[i], label='train')
        plot_stats(history, 'valid_' + loss_term, True, axes[i], label='valid')
        axes[i].set_title(loss_term)
    file = join(path, 'loss_terms.pdf')
    _save_fig(fig, file)

def save_batch_stats_fig(history, keys, path, fig_name):
    if not isinstance(keys, list):
        keys = [keys]
    fig, ax = plt.subplots()
    for key in keys:
        plot_stats(history, key, True, ax, label=key)
    ax.legend()
    file = join(path, fig_name + '.pdf')
    _save_fig(fig, file)

def create_rectangle(box, ec='r', lw=1):
    from matplotlib.patches import Rectangle
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()
    x, y, x2, y2 = tuple(box)
    w = x2 - x
    h = y2 - y
    return Rectangle((x, y), w, h, fill=False, ec=ec, lw=lw)

class MultiOptimLRScheduler(Callback):
    '''
    Source: https://github.com/skorch-dev/skorch/blob/fc3758b/skorch/callbacks/lr_scheduler.py
    '''
    def __init__(self,
        policies,
        event_name='event_lr',
        **kwargs
    ):
        self.policies = policies if isinstance(policies, list) else [policies]
        self.event_name = event_name
        vars(self).update(kwargs)

    def initialize(self):
        self.lr_schedulers_ = []
        return self

    def _get_policy_cls(self, policy):
        if isinstance(policy, str):
            return getattr(sys.modules[__name__], policy)
        return policy

    def get_kwargs(self, i):
        excluded = ('policies', 'event_name')
        kwargs = {}
        for k, v in vars(self).items():
            if not (k in excluded or k.endswith('_')):
                kwargs[k] = v[i] if isinstance(v, list) else v
        return kwargs

    def on_train_begin(self, net, **kwargs):
        if len(self.policies) == 1 and len(net._optimizers) > 1:
            self.policies = len(net._optimizers) * self.policies
        assert len(self.policies) == len(net._optimizers)
        for i, name in enumerate(net._optimizers):
            optimizer = getattr(net, name + '_')
            policy = self._get_policy_cls(self.policies[i])
            kwargs = self.get_kwargs(i)
            self.lr_schedulers_ += [
                self._get_scheduler(
                    optimizer, policy, **kwargs
                )
            ]

    def on_epoch_end(self, net, **kwargs):
        for i, lr_scheduler in enumerate(self.lr_schedulers_):
            if self.event_name is not None and hasattr(
                lr_scheduler, 'get_last_lr'
            ):
                net.history.record(
                    self.event_name + str(i+1),
                    lr_scheduler.get_last_lr()[0]
                )
            lr_scheduler.step()

    def _get_scheduler(self, optimizer, policy, **scheduler_kwargs):
        if 'last_epoch' not in scheduler_kwargs:
            scheduler_kwargs['last_epoch'] = -1
        return policy(optimizer, **scheduler_kwargs)

class MyDecay(tensorflow.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, mu_0=0.01, alpha=10, beta=0.75, max_steps=1000):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.max_steps = float(max_steps)

    def __call__(self, step):
        step = tensorflow.cast(step, tensorflow.float32)
        p = step / self.max_steps
        return self.mu_0 / (1+self.alpha * p)**self.beta

    def get_config(self):
        config = {
            'mu_0': self.mu_0,
            'alpha': self.alpha,
            'beta': self.beta,
            'max_steps': self.max_steps
        }
        return config

def collect_dalupi_probas_and_outputs(net, X):
    from .models import DALUPI
    assert isinstance(net, DALUPI)
    dataset = net.get_dataset(X)
    iterator = net.get_iterator(dataset, training=False)
    probas, outputs = [], []
    for batch in iterator:
        probas_, output = net.evaluation_step(batch, training=False, return_outputs=True)
        probas += [probas_]
        outputs += output
    probas = torch.cat(probas).numpy()
    return probas, outputs

def collect_inputs_and_targets(net, X):
    dataset = net.get_dataset(X)
    X, y = [], []
    for batch in net.get_iterator(dataset, training=False):
        inputs, targets = unpack_data(batch)
        for i, t in zip(inputs, targets):
            X.append(i)
            y.append(t)
    return X, y

def compute_saliency_maps(net, X, indices=None):
    dataset = net.get_dataset(X)
    iterator = net.get_iterator(dataset, training=False)
    saliency_maps = []
    i = 0
    for batch in iterator:
        Xi, _ = unpack_data(batch)
        for xi in Xi:
            xi.requires_grad_() # Compute gradients...
            net._set_training(training=False)  # ... but run model in evaluation mode
            output = net.infer(xi)
            j = output.argmax() if indices is None else indices[i]
            selected_output = output[0, j]
            selected_output.backward()
            saliency_maps += [xi.grad]
            i += 1
    return saliency_maps
