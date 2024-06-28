import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from config import cfg
from my_functionals.DualTaskLoss import DualTaskLoss

def get_loss(args):
    """
    Configure and return the loss function based on the specified arguments.
    Supports the selection between different loss functions based on input.

    Args:
        args: An object containing configurations and parameters for loss function selection.

    Returns:
        criterion: The configured loss function for training.
        criterion_val: The configured loss function for validation.
    """
    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes, reduction='mean',
            ignore_index=args.dataset_cls.ignore_label, 
            upper_bound=args.wt_bound).cuda()
    elif args.joint_edgeseg_loss:
        criterion = JointEdgeSegLoss(classes=args.dataset_cls.num_classes,
           ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
           edge_weight=args.edge_weight, seg_weight=args.seg_weight, att_weight=args.att_weight, dual_weight=args.dual_weight).cuda()
    else:
        criterion = CrossEntropyLoss2d(reduction='mean',
                                       ignore_index=args.dataset_cls.ignore_label).cuda()

    criterion_val = JointEdgeSegLoss(classes=args.dataset_cls.num_classes, mode='val',
       ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
       edge_weight=args.edge_weight, seg_weight=args.seg_weight).cuda()

    return criterion, criterion_val

class JointEdgeSegLoss(nn.Module):
    """
    Implements a composite loss function for semantic segmentation and edge detection.
    Combines segmentation and edge loss with configurable weights.
    """
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0, mode='train', 
                 edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if mode == 'train':
            self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
        elif mode == 'val':
            self.seg_loss = CrossEntropyLoss2d(reduction='mean',
                                               ignore_index=ignore_index).cuda()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight
        self.dual_task = DualTaskLoss()


    def bce2d(self, input, target):
        """
        Calculates a weighted binary cross-entropy loss for edge detection. The weight for each pixel is inversely 
        proportional to the frequency of its class, mitigating class imbalance. Pixels marked to be ignored are 
        excluded from the loss calculation.

        Parameters:
        - input (torch.Tensor): The predicted logits for edge detection, with dimensions [N, C, H, W], where N is the 
          batch size, C is the number of channels (usually 1 for edge detection), H is the image height, and W is the 
          image width.
        - target (torch.Tensor): The ground-truth edge map, with dimensions [N, H, W], where N is the batch size, 
          H is the image height, and W is the image width.

        Returns:
        - torch.Tensor: The calculated binary cross-entropy loss as a scalar tensor.
        """
        n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t ==1)
        neg_index = (target_t ==0)
        ignore_index=(target_t >1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction = 'mean')
        return loss

    def edge_attention(self, input, target, edge):
        """
        Applies an attention mechanism to the segmentation task based on edge predictions. Segmentation predictions 
        are adjusted where edge confidence exceeds a threshold, focusing the segmentation loss on areas near detected 
        edges.

        Parameters:
        - input (torch.Tensor): The predicted logits for segmentation, with dimensions [N, C, H, W].
        - target (torch.Tensor): The ground-truth segmentation map, with dimensions [N, H, W].
        - edge (torch.Tensor): The predicted logits for edge detection, used to apply attention, with dimensions 
          [N, C, H, W].

        Returns:
        - torch.Tensor: The attention-adjusted segmentation loss as a scalar tensor.
        """
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input, torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets):
        """
        Computes the total loss by combining the segmentation loss, edge detection loss, attention mechanism loss, 
        and dual-task loss, each weighted by their respective weights. The losses are calculated based on the 
        inputs and targets provided for both segmentation and edge detection tasks.

        Parameters:
        - inputs (tuple[torch.Tensor, torch.Tensor]): A tuple containing the predicted logits for segmentation and 
          edge detection, each with dimensions [N, C, H, W].
        - targets (tuple[torch.Tensor, torch.Tensor]): A tuple containing the ground-truth maps for segmentation and 
          edge detection, each with dimensions [N, H, W].

        Returns:
        - dict[str, torch.Tensor]: A dictionary of the computed losses, including 'seg_loss', 'edge_loss', 
          'att_loss', and 'dual_loss'.
        """
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        losses['edge_loss'] = self.edge_weight * 20 * self.bce2d(edgein, edgemask)
        losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)
        losses['dual_loss'] = self.dual_weight * self.dual_task(segin, segmask)
              
        return losses

class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    A custom loss function for semantic segmentation that dynamically calculates class weights
    based on their frequency within each image, to mitigate the class imbalance problem.
    """
    def __init__(self, classes, weight=None, ignore_index=255, norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(weight=None, reduction='mean', ignore_index=ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), density=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        """
        Calculates the loss for a batch of inputs and targets, applying dynamically calculated weights.
        """
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1), targets[i].unsqueeze(0))
        return loss

class CrossEntropyLoss2d(nn.Module):
    """
    Implements a 2D version of the cross-entropy loss for use in semantic segmentation tasks.
    Combines log softmax and negative log likelihood (NLL) loss in a single class.
    """
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        """
        Computes the cross-entropy loss for the provided inputs and targets.
        """
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
