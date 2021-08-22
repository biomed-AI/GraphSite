from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse

NUM_CLASSES = 1

# hard example
def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

def get_hard_samples(logits,labels,neg_more=2,neg_least_ratio=0.5,neg_max_ratio=0.7):
    logits = logits.view(-1)
    labels = labels.view(-1)

    pos_idcs = labels > 0.6
    pos_output = logits[pos_idcs]
    pos_labels = labels[pos_idcs]

    neg_idcs = labels <= 0.4
    neg_output = logits[neg_idcs]
    neg_labels = labels[neg_idcs]

    neg_at_least=max(neg_more,int(neg_least_ratio * neg_output.size(0)))
    hard_num = min(neg_output.size(0),pos_output.size(0) + neg_at_least, int(neg_max_ratio * neg_output.size(0)) + neg_more)
    if hard_num > 0:
        neg_output, neg_labels = hard_mining(neg_output, neg_labels, hard_num)

    logits=torch.cat([pos_output,neg_output])
    labels = torch.cat([pos_labels, neg_labels])


    return logits,labels

# https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------

def binary_lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss



def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target, epoch=0):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()

class HardLogLoss(nn.Module):
    def __init__(self):
        super(HardLogLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.__classes_num = NUM_CLASSES

    def forward(self, logits, labels,epoch=0):
        labels = labels.float()
        loss=0
        for i in range(NUM_CLASSES):
            logit_ac=logits[:,i]
            label_ac=labels[:,i]
            logit_ac, label_ac=get_hard_samples(logit_ac,label_ac)
            loss+=self.bce_loss(logit_ac,label_ac)
        loss = loss/NUM_CLASSES
        return loss

# https://github.com/bermanmaxim/LovaszSoftmax/tree/master/pytorch
def lovasz_hinge(logits, labels, ignore=None, per_class=True):
    """
    Binary Lovasz hinge loss
      logits: [B, C] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, C] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_class:
        loss = 0
        for i in range(NUM_CLASSES):
            logit_ac = logits[:, i]
            label_ac = labels[:, i]
            loss += lovasz_hinge_flat(logit_ac, label_ac)
        loss = loss / NUM_CLASSES
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        loss = lovasz_hinge_flat(logits, labels)
    return loss

# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053
class SymmetricLovaszLoss(nn.Module):
    def __init__(self):
        super(SymmetricLovaszLoss, self).__init__()
        self.__classes_num = NUM_CLASSES

    def forward(self, logits, labels,epoch=0):
        labels = labels.float()
        loss=((lovasz_hinge(logits, labels)) + (lovasz_hinge(-logits, 1 - labels))) / 2
        return loss

class FocalSymmetricLovaszHardLogLoss(nn.Module):
    def __init__(self):
        super(FocalSymmetricLovaszHardLogLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.slov_loss = SymmetricLovaszLoss()
        self.log_loss = HardLogLoss()
    def forward(self, logit, labels,epoch=0):
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        slov_loss = self.slov_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss*0.5 + slov_loss*0.5 +log_loss * 0.5
        return loss

# https://github.com/ronghuaiyang/arcface-pytorch
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss


class PairLoss(nn.Module):
    def __init__(self,margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, feature,label):
        n,m = feature.shape
        k = label.shape[1]
        feature0 = feature.repeat(n,1)
        feature1 = feature.repeat(1,n).view(-1,m)
        label0 = label.repeat(n,1)
        label1 = label.repeat(1,n).view(-1,k)

        # loss = ((label0==label1).float().min(dim=1,keepdim=True)[0]*2-1) * torch.norm(feature0-feature1,keepdim=True,dim=[1]) / np.sqrt(m)
        # print(((label0==label1)&(label0>0.5)).float().sum(dim=1,keepdim=True))
        # print(((label0>0.5)|(label1>0.5)).float().sum(dim=1,keepdim=True))
        weight = ((label0==label1)&(label0>0.5)).float().sum(dim=1,keepdim=True) / ((label0>0.5)|(label1>0.5)).float().sum(dim=1,keepdim=True)
        weight[weight==0] = -1.0
        # loss = weight * torch.norm(feature0-feature1,keepdim=True,dim=[1]) / np.sqrt(m)
        # print(torch.clamp((feature0-feature1).pow(2).sqrt().sum(dim=1,keepdim=True)/m,min=None,max=self.margin))
        dist = torch.norm(feature0-feature1,keepdim=True,dim=[1]) / np.sqrt(m)
        max_dist = torch.zeros_like(dist) + self.margin
        loss = weight * torch.minimum(dist,max_dist)
        # print(loss)
        return loss.mean()
