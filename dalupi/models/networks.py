import torch
import torchvision
import torchvision.models.detection as D
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from dalupi.models.utils import make_accept_grayscale
from torchvision.ops import boxes as box_ops
from typing import Dict, List
from skorch.utils import params_for
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm

MAX_N_TRAINABLE_RESNET_LAYERS = 5

def get_adapt_encoder(weights=None, n_trainable_layers=5, dropout=0, **kwargs):
    resnet = ResNet50(include_top=False, weights=weights, pooling='avg', **kwargs)
    
    trainable_layers = [5, 4, 3, 2, 1][:n_trainable_layers]
    trainable_layers = ['conv%d' % l for l in trainable_layers]
    if n_trainable_layers > 0:
        trainable_layers += ['avg']
    if n_trainable_layers == 5:
        trainable_layers += ['input', 'pool1']
    for layer in resnet.layers:
        layer.trainable = layer.name.split('_')[0] in trainable_layers
    
    if weights is not None:
        for i in range(len(resnet.layers)):
            if resnet.layers[i].__class__.__name__ == 'BatchNormalization':
                resnet.layers[i].trainable = False
    
    model = Sequential()
    model.add(resnet)
    model.add(Dropout(dropout))
    
    return model

def get_adapt_task(
    in_features,
    out_features,
    is_nonlinear,
    dropout=0,
    out_activation='linear'
):
    model = Sequential()
    if is_nonlinear:
        model.add(Dense(in_features // 2, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(in_features // 4, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(out_features, activation=out_activation))
    return model

def _get_dense_norm(out_features, activation, max_norm):
    return Dense(
        out_features,
        activation=activation,
        kernel_constraint=MaxNorm(max_norm),
        bias_constraint=MaxNorm(max_norm)
    )

def get_adapt_task_norm(
    in_features,
    out_features,
    is_nonlinear,
    dropout=0,
    max_norm=0.5,
    out_activation='linear'
):
    model = Sequential()
    if is_nonlinear:
        model.add(_get_dense_norm(in_features // 2, 'relu', max_norm))
        model.add(Dropout(dropout))
        model.add(_get_dense_norm(in_features // 4, 'relu', max_norm))
        model.add(Dropout(dropout))
    model.add(_get_dense_norm(out_features, out_activation, max_norm))
    return model

def get_adapt_discriminator(depth, width=64, n_domains=2, dropout=0):
    model = Sequential()
    for _ in range(depth - 1):
        model.add(Dense(width, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(n_domains - 1, activation='sigmoid'))
    return model

class ConvNet(torch.nn.Module):
    def __init__(self, out_features=5, dropout_rate=0.4, dropout_rate2=0.2, l1=50):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.drop1 = nn.Dropout(p=dropout_rate)  
        self.fc1 = nn.Linear(32 * 7 * 7, l1)
        self.drop2 = nn.Dropout(p=dropout_rate2)
        self.fc2 = nn.Linear(l1, out_features)
        self.out_features = out_features

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

def get_trainable_resnet_layers(n_trainable_layers):
    all_layers = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']
    if n_trainable_layers < 0 or n_trainable_layers > 5:
        raise ValueError(
            f'n_trainable_layers should be in the range [0, 5], got {n_trainable_layers}.'
        )
    trainable_layers = all_layers[:n_trainable_layers]
    if n_trainable_layers == 5:
        trainable_layers.append('bn1')
    return trainable_layers

class ResNet(torch.nn.Module):
    def __init__(self, version, grayscale_input, pretrained=True, n_trainable_layers=0, p_dropout=0):
        super().__init__()

        self.pretrained = pretrained
        
        weights = 'DEFAULT' if pretrained else None
        norm_layer = torchvision.ops.misc.FrozenBatchNorm2d if pretrained else torch.nn.BatchNorm2d

        if version == 18:
            self.network = torchvision.models.resnet18(weights=weights, norm_layer=norm_layer)
            self.out_features = 512
        elif version == 50:
            self.network = torchvision.models.resnet50(weights=weights, norm_layer=norm_layer)
            self.out_features = 2048
        else:
            raise ValueError('version must be either 18 or 50.')
        
        if not pretrained:
            n_trainable_layers = MAX_N_TRAINABLE_RESNET_LAYERS
        
        self.trainable_layers = get_trainable_resnet_layers(n_trainable_layers)

        if grayscale_input:
            requires_grad = n_trainable_layers == MAX_N_TRAINABLE_RESNET_LAYERS
            self.network.conv1 = make_accept_grayscale(self.network.conv1, requires_grad)
                
        self.dropout = torch.nn.Dropout(p_dropout)

        del self.network.fc
        self.network.fc = Identity()
        
    def forward(self, x):
        return self.dropout(self.network(x))

class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features, width, depth, p_dropout=0):
        super(MLP, self).__init__()
        self.input_layer = torch.nn.Linear(in_features, width)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(width, width) for _ in range(depth - 2)]
        )
        self.output_layer = torch.nn.Linear(width, out_features)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Task(torch.nn.Module):
    def __init__(self, in_features, out_features, is_nonlinear=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if is_nonlinear:
            self.task = torch.nn.Sequential(
                torch.nn.Linear(in_features, in_features // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 2, in_features // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 4, out_features)
            )
        else:
            self.task = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.task(x)

class BaseNetwork(torch.nn.Module):
    def __init__(self, featurizer, task, **kwargs):
        super().__init__()
        
        featurizer_kwargs = params_for('featurizer', kwargs)
        self.featurizer = featurizer(**featurizer_kwargs)

        task_in_features = self.featurizer.out_features
        task_kwargs = params_for('task', kwargs)
        self.task = task(task_in_features, **task_kwargs)

    def forward(self, x):
        return self.task(self.featurizer(x))

    def _freeze(net, x):
        self = net.module_
        if not hasattr(self.featurizer, 'trainable_layers'):
            return False
        if x.startswith('featurizer'):
            layer = x.split('.')[2]
            return layer not in self.featurizer.trainable_layers
        else:
            return False

class DANN(torch.nn.Module):
    def __init__(self, featurizer, classifier, discriminator, conditional=False, class_balance=False, **kwargs):
        super().__init__()

        self.conditional = conditional
        self.class_balance = class_balance

        featurizer_kwargs = params_for('featurizer', kwargs)
        self.featurizer = featurizer(**featurizer_kwargs)

        classifier_in_features = self.featurizer.out_features
        classifier_kwargs = params_for('classifier', kwargs)
        self.classifier = classifier(classifier_in_features, **classifier_kwargs)

        discriminator_in_features = self.featurizer.out_features
        discriminator_kwargs = params_for('discriminator', kwargs)
        self.discriminator = discriminator(discriminator_in_features, **discriminator_kwargs)

        self.class_embeddings = torch.nn.Embedding(
            self.classifier.out_features,
            self.featurizer.out_features
        )

    def forward(self, x, y, infer_classifier, classifier_loss_fcn):       
        disc_labels = y[:, -1].long()
        if self.training:
            disc_labels_np = disc_labels.cpu().numpy()
            assert set(disc_labels_np) == {0, 1}, disc_labels_np

        y = y[:, :-1]
        z = self.featurizer(x)
        if self.conditional:
            disc_input = z + self.class_embeddings(y)
        else:
            disc_input = z
        disc_output = self.discriminator(disc_input)
        
        if self.class_balance:
            raise NotImplementedError
        else:
            disc_loss = F.cross_entropy(disc_output, disc_labels)
        
        loss_dict = {'disc_loss': disc_loss}

        if self.training:
            input_grad = torch.autograd.grad(
                F.cross_entropy(disc_output, disc_labels, reduction='sum'),
                [disc_input],
                create_graph=True
            )[0]
            grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
            loss_dict['grad_penalty'] = grad_penalty
        
        y_pred = None

        if infer_classifier:
            # FIXME: Should not classify target samples
            y_pred = self.classifier(z)
            unreduced_classifier_loss = classifier_loss_fcn(y_pred, y)
            y_mask = disc_labels == 0  # the source domain has label 0
            classifier_loss = unreduced_classifier_loss[y_mask].mean()
            loss_dict['classifier_loss'] = classifier_loss
        
        return loss_dict, y_pred
    
    def predict(self, x):
        return self.classifier(self.featurizer(x))
    
    def _freeze(net, x):
        if x.startswith('featurizer'):
            self = net.module_
            layer = x.split('.')[2]
            return layer not in self.featurizer.trainable_layers
        else:
            return False

class DALUPI(torch.nn.Module):
    def __init__(
        self,
        out_features,
        backbone_weights,
        backbone_n_trainable_layers,
        backbone_grayscale_input,
        rpn_anchor_sizes,
        rpn_aspect_ratios,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        rpn_score_thresh,
        box_roi_pool_output_size,
        box_roi_pool_sampling_ratio,
        box_head_output_size,
        box_fg_iou_thresh,
        box_bg_iou_thresh,
        box_batch_size_per_image,
        box_positive_fraction,
        box_bbox_reg_weights,
        box_score_thresh,
        box_nms_thresh,
        box_detections_per_img,
        transform_min_size,
        transform_max_size,
        transform_image_mean,
        transform_image_std,
        **kwargs
    ):
        super().__init__()

        self.out_features = out_features
        
        # Backbone
        assert backbone_weights in ['COCO_V1', 'IMAGENET1K_V2', None]
        weights = 'DEFAULT' if backbone_weights == 'COCO_V1' else None
        weights_backbone = 'DEFAULT' if backbone_weights == 'IMAGENET1K_V2' else None
        backbone = D.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            trainable_backbone_layers=backbone_n_trainable_layers
        ).backbone
        if backbone_grayscale_input:
            requires_grad = backbone_n_trainable_layers == MAX_N_TRAINABLE_RESNET_LAYERS
            backbone.body['conv1'] = make_accept_grayscale(
                backbone.body['conv1'],
                requires_grad
            )
        
        # RPN
        if len(rpn_anchor_sizes) > len(rpn_aspect_ratios):
            rpn_aspect_ratios = rpn_aspect_ratios * len(rpn_anchor_sizes)
        else:
            assert len(rpn_anchor_sizes) == len(rpn_aspect_ratios)
        rpn_anchor_generator = D.anchor_utils.AnchorGenerator(
            sizes=rpn_anchor_sizes,
            aspect_ratios=rpn_aspect_ratios
        )
        assert len(set(rpn_anchor_generator.num_anchors_per_location())) == 1
        n_anchors = rpn_anchor_generator.num_anchors_per_location()[0]
        rpn_head = D.rpn.RPNHead(backbone.out_channels, n_anchors)
        rpn_kwargs = {
            'anchor_generator': rpn_anchor_generator,
            'head': rpn_head,
            'fg_iou_thresh': rpn_fg_iou_thresh,
            'bg_iou_thresh': rpn_bg_iou_thresh,
            'batch_size_per_image': rpn_batch_size_per_image,
            'positive_fraction': rpn_positive_fraction,
            'pre_nms_top_n': rpn_pre_nms_top_n,
            'post_nms_top_n': rpn_post_nms_top_n,
            'nms_thresh': rpn_nms_thresh,
            'score_thresh': rpn_score_thresh
        }
        rpn = RPN(**rpn_kwargs)

        # RoI heads
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=box_roi_pool_output_size,
            sampling_ratio=box_roi_pool_sampling_ratio
        )
        box_head = D.faster_rcnn.TwoMLPHead(
            backbone.out_channels * box_roi_pool_output_size ** 2,
            box_head_output_size
        )
        box_predictor = D.faster_rcnn.FastRCNNPredictor(box_head_output_size, out_features)
        box_kwargs = {
            'box_roi_pool': box_roi_pool,
            'box_head': box_head,
            'box_predictor': box_predictor,
            'fg_iou_thresh': box_fg_iou_thresh,
            'bg_iou_thresh': box_bg_iou_thresh,
            'batch_size_per_image': box_batch_size_per_image,
            'positive_fraction': box_positive_fraction,
            'bbox_reg_weights': box_bbox_reg_weights,
            'score_thresh': box_score_thresh,
            'nms_thresh': box_nms_thresh,
            'detections_per_img': box_detections_per_img
        }
        roi_heads = RoIHeads(**box_kwargs)

        # Transform
        transform = D.transform.GeneralizedRCNNTransform(
            transform_min_size,
            transform_max_size,
            transform_image_mean,
            transform_image_std, 
            **kwargs
        )

        self.model = D.generalized_rcnn.GeneralizedRCNN(backbone, rpn, roi_heads, transform)
        
    def print_named_tensors(self):
        for parameter_name, tensor in self.model.named_parameters():
            print(parameter_name, tensor.requires_grad)

    def forward(self, **inputs):
        return self.model(**inputs)  # model takes "images" and "targets"
    
    def _freeze(net, x):
        # The backbone is frozen upon calling _resnet_fpn_extractor
        pass

class GeneralizedRCNN(D.generalized_rcnn.GeneralizedRCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, images, targets=None):
        import warnings
        from typing import List, Tuple
        from collections import OrderedDict
        
        if self.training:
            if targets is None:
                torch._assert(False, 'targets should not be none when in training mode')
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f'Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.',
                        )
                    else:
                        torch._assert(False, f'Expected target boxes to be of type Tensor, got {type(boxes)}.')

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f'expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}',
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target['boxes']
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # Print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        'All bounding boxes should have positive height and width.'
                        f' Found invalid box {degen_bb} for target at index {target_idx}.',
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn('RCNN always returns a (Losses, Detections) tuple in scripting')
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

class RPN(D.rpn.RegionProposalNetwork):
    '''
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_batch_proposals = []
    
    def forward(self, images, features, targets):
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = D.rpn.concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # Apply pred_bbox_deltas to anchors to obtain the decoded proposals.
        # Note that we detach the deltas because Faster R-CNN does not backprop through
        # the proposals.
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        self.n_batch_proposals += [sum([b.shape[0] for b in boxes])]

        losses = {}
        if self.training:
            if targets is None:
                raise ValueError('targets should not be None')
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            loss_mask = [t['box_loss_mask'].item() for t in targets]
            for i in range(len(labels)):
                if loss_mask[i] == 0:
                    # The sampler in self.compute_loss ignores -1 labels so the masks
                    # for positive and negative proposals in this image are both a zero tensor.
                    # Thus, no proposals from this image will be considered in the loss computation.
                    labels[i][:] = -1
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                'loss_objectness': loss_objectness,
                'loss_rpn_box_reg': loss_rpn_box_reg,
            }
        return boxes, losses

def fastrcnn_loss(
    class_logits,
    box_regression,
    labels,
    regression_targets,
    cls_loss_mask,
    box_loss_mask,
):
    '''
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
    '''
    def convert_labels(labels, n_classes, n_labels_per_image, device):
        '''
        Convert labels on the form [1, 3] to a binary vector.
        Labels are assumed to start from 0. For example, with n_classes = 4, 
        the label [1, 3] will be converted to [0, 1, 0, 1].
        '''
        out = torch.zeros((len(labels), n_classes), device=device)
        ii = np.repeat(range(len(labels)), n_labels_per_image)
        jj = torch.cat(labels)
        out[ii, jj] = 1
        return out

    # First, we compute the classification loss for non-background examples
    # from the source domain for which there are NO bounding boxes (mask -1).
    # Since we do not have any boxes, we collect the highest score for each
    # class and compute a multilabel classification loss (BCE loss).

    manipulated_labels = []
    labels_fg_no_pi = []
    n_labels_per_image = []
    chunk_sizes = []
    for labels_per_image in labels:
        if labels_per_image.dim() == 2:
            manipulated_labels.append(
                torch.zeros(
                    labels_per_image.shape[0],
                    dtype=labels_per_image.dtype,
                    device=labels_per_image.device
                )
            )
            if labels_per_image.shape[0] == 0:
                # There were no proposals in the image.
                continue
            labels_fg_no_pi.append(labels_per_image[0])  # all labels are the same
            n_labels_per_image.append(labels_per_image.shape[1])
            # We do not want to sample any proposals from this image for
            # bounding box loss computation. We fix this by using
            # a zero tensor as labels for this image.
            chunk_sizes.append(labels_per_image.shape[0])
        else:
            manipulated_labels.append(labels_per_image)

    mask = cls_loss_mask == -1
    if torch.any(mask):
        class_logits_fg_no_pi = class_logits[mask]
        class_logits_fg_no_pi = torch.split(class_logits_fg_no_pi, chunk_sizes)
        class_logits_fg_no_pi = torch.nn.utils.rnn.pad_sequence(class_logits_fg_no_pi, batch_first=True)
        class_logits_fg_no_pi = torch.max(class_logits_fg_no_pi, dim=1).values  # take max over proposals
        
        labels_fg_no_pi = convert_labels(labels_fg_no_pi, class_logits.shape[1], n_labels_per_image, class_logits.device)
        
        class_softmax_fg_no_pi = F.softmax(class_logits_fg_no_pi, dim=1)
        bce_loss = F.binary_cross_entropy(class_softmax_fg_no_pi, labels_fg_no_pi)
    else:
        bce_loss = torch.tensor(float('nan'), device=class_logits.device)
    
    # Then, we compute the classification loss for (non-background) examples
    # from the source domain for which there ARE bounding boxes and for
    # background examples from the source domain (mask 1).

    labels = torch.cat(manipulated_labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    mask = cls_loss_mask == 1
    ce_loss = F.cross_entropy(class_logits[mask], labels[mask])

    # Finally, we compute the regression loss for examples with bounding boxes.
    # Note that we have modified the labels so that examples without bounding
    # boxes have a zero array as labels. Because we only sample examples with 
    # positive labels, we will not include any images without PI.

    box_loss_mask_pos_inds = torch.where(box_loss_mask)[0]

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    assert all([i in box_loss_mask_pos_inds for i in sampled_pos_inds_subset]) 
    labels_pos = labels[sampled_pos_inds_subset]
    box_regression = box_regression.reshape(class_logits.shape[0], box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1/9,
        reduction='sum'
    )
    box_loss = box_loss / labels.numel()  # should we change the denominator?

    mask = cls_loss_mask > -1
    n_pos_labels = (labels[mask] > 0).sum().cpu().item()
    n_neg_labels = (labels[mask] == 0).sum().cpu().item()

    return bce_loss, ce_loss, box_loss, n_pos_labels, n_neg_labels

class RoIHeads(D.roi_heads.RoIHeads):
    '''
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_batch_positives = []
        self.n_batch_negatives = []
    
    def get_loss_masks(self, targets, labels, device):
        cls_loss_mask, box_loss_mask = [], []
        for t, l in zip(targets, labels):
            n = l.shape[0]
            cls_loss_mask.append(torch.full((n,), t['cls_loss_mask'], device=device))
            box_loss_mask.append(torch.full((n,), t['box_loss_mask'], device=device))
        return torch.cat(cls_loss_mask), torch.cat(box_loss_mask)
    
    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # The sampler has identified an arbitrary number of positive proposals in images
        # from the source domain for which there are no bounding boxes. Since the labels for
        # these images are 2-D tensors, the number of sampled indices does not equal the
        # desired batch size per image. We resample proposals here.
        
        for i, labels_per_image in enumerate(labels):
            if labels_per_image.dim() == 2:
                sampled_pos_inds[i] = torch.zeros(labels_per_image.shape[0], dtype=torch.uint8)
                # labels_per_image.shape[0] equals the number of proposals in image i
                ii = torch.randperm(labels_per_image.shape[0])[:self.fg_bg_sampler.batch_size_per_image]
                sampled_pos_inds[i][ii] = 1
                sampled_neg_inds[i] = torch.zeros(labels_per_image.shape[0], dtype=torch.uint8)

        sampled_inds = []
        for pos_inds_img, neg_inds_img in zip(sampled_pos_inds, sampled_neg_inds):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds
    
    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, cls_loss_mask):
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, image_mask in zip(proposals, gt_boxes, gt_labels, cls_loss_mask):
            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                if image_mask.item() == 1:
                    # Background image
                    labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                else:
                    # Non-background image for which there is no bounding box
                    assert image_mask.item() == -1
                    # Note: the sampler assumes a 1-D tensor, but it will work also for a 2-D tensor.
                    # We let labels_in_image.shape[0] be equal to the number of proposals in the image,
                    # so that we can resample proposals later (we do not care about positive and negative
                    # proposals).
                    labels_in_image = gt_labels_in_image.expand(proposals_in_image.shape[0], -1).to(dtype=torch.int64)
            else:
                # Set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)  # Assign a gt box to each proposal

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def select_training_samples(
        self,
        proposals,
        targets,
    ):
        self.check_targets(targets)
        if targets is None:
            raise ValueError('targets should not be None')
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t['boxes'].to(dtype) for t in targets]
        gt_labels = [t['labels'] for t in targets]
        cls_loss_mask = [t['cls_loss_mask'] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, cls_loss_mask)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def forward(
        self,
        features,
        proposals,
        image_shapes,
        targets=None,
    ):
        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
               
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)  # box_features: (n_proposals, representation_size)
        class_logits, box_regression = self.box_predictor(box_features)  # class_logits: (n_proposals, n_classes); box_regression; (n_proposals, 4*n_classes)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError('labels cannot be None')
            if regression_targets is None:
                raise ValueError('regression_targets cannot be None')
            device = class_logits.device
            cls_loss_mask, box_loss_mask = self.get_loss_masks(targets, labels, device)
            bce_loss, ce_loss, loss_box_reg, n_pos_labels, n_neg_labels = fastrcnn_loss(
                class_logits,
                box_regression,
                labels,
                regression_targets,
                cls_loss_mask,
                box_loss_mask
            )
            self.n_batch_positives += [n_pos_labels]
            self.n_batch_negatives += [n_neg_labels]
            losses = {'bce_loss_classifier': bce_loss, 'ce_loss_classifier': ce_loss, 'loss_box_reg': loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({'boxes': boxes[i], 'labels': labels[i], 'scores': scores[i]})
        
        assert not self.has_mask()
        assert self.keypoint_roi_pool is None and self.keypoint_head is None and self.keypoint_predictor is None

        return result, losses
