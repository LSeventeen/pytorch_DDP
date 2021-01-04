import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lovasz_losses import lovasz_softmax


def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1., ignore_index=255):
#         super(DiceLoss, self).__init__()
#         self.ignore_index = ignore_index
#         self.smooth = smooth
#
#     def forward(self, output, target):
#         if self.ignore_index not in range(target.min(), target.max()):
#             if (target == self.ignore_index).sum() > 0:
#                 target[target == self.ignore_index] = target.min()
#         target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
#         output = F.softmax(output, dim=1)
#         output_flat = output.contiguous().view(-1)
#         target_flat = target.contiguous().view(-1)
#         intersection = (output_flat * target_flat).sum()
#         loss = 1 - ((2. * intersection + self.smooth) /
#                     (output_flat.sum() + target_flat.sum() + self.smooth))
#         return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE_loss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# class CE_DiceLoss(nn.Module):
#     def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
#         super(CE_DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.dice = DiceLoss()
#         self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
#
#     def forward(self, output, target):
#         CE_loss = self.cross_entropy(output, target)
#         dice_loss = self.dice(output, target)
#         return CE_loss + dice_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss





class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.bce_loss = nn.L1Loss()

    def forward(self, prediction, targets):
        prediction_flat = prediction.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(prediction_flat, targets_flat)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.bce_loss = nn.MSELoss()

    def forward(self, prediction, targets):
        prediction_flat = prediction.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(prediction_flat, targets_flat)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, prediction, targets):
        prediction_flat = prediction.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(prediction_flat, targets_flat)


class BCELoss_2(nn.Module):
    def __init__(self, weight=0.5):
        super(BCELoss_2, self).__init__()
        self.bce_loss = BCELoss()
        self.weight = weight

    def forward(self, prediction, gt, prediction_s, gt_s):
        return (1 - self.weight) * self.bce_loss(prediction_s, gt_s) + self.weight * self.bce_loss(prediction, gt)


class BCELoss_3(nn.Module):
    def __init__(self, weight=0.5):
        super(BCELoss_3, self).__init__()
        self.bce_loss = BCELoss()
        self.weight = weight

    def forward(self, prediction, gt, prediction_s, gt_s, prediction_l, gt_l):
        return self.weight / 2 * self.bce_loss(prediction_s,
                                               gt_s) + self.weight / 2 * self.bce_loss(
            prediction_l, gt_l) + self.weight * self.bce_loss(prediction, gt)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        # target_flat = target.contiguous().view(-1)
        # intersection = (output_flat * target_flat).sum()
        # loss = 1 - ((2. * intersection + self.smooth) /
        #             (output_flat.sum() + target_flat.sum() + self.smooth))
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss


class DiceLoss_2(nn.Module):
    def __init__(self, weight=0.5):
        super(DiceLoss_2, self).__init__()
        self.DiceLoss = DiceLoss(smooth=1e-8)
        self.weight = weight

    def forward(self, prediction, gt, prediction_s, gt_s):
        return (1 - self.weight) * self.DiceLoss(prediction_s, gt_s) + 1 - self.weight * self.DiceLoss(
            prediction, gt)


class DiceLoss_3(nn.Module):
    def __init__(self, weight=0.5):
        super(DiceLoss_3, self).__init__()
        self.DiceLoss = DiceLoss(smooth=1e-8)
        self.weight = weight

    def forward(self, prediction, gt, prediction_s, gt_s, prediction_l, gt_l):
        return self.weight / 2 * self.DiceLoss(prediction_l, gt_l) + self.weight / 2 * self.DiceLoss(
            prediction_s, gt_s) + + self.weight * self.DiceLoss(prediction, gt)


class CE_DiceLoss(nn.Module):
    def __init__(self, D_weight=0.5):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss()
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)


class CE_DiceLoss_2(nn.Module):
    def __init__(self, weight=0.5, D_weight=0.5):
        super(CE_DiceLoss_2, self).__init__()
        self.CE_DiceLoss = CE_DiceLoss(D_weight=D_weight)
        self.weight = weight

    def forward(self, prediction, targets, prediction_small, targets_small):
        return self.weight * self.CE_DiceLoss(prediction, targets) + (1 - self.weight) * self.CE_DiceLoss(
            prediction_small, targets_small)

class CE_DiceLoss3(nn.Module):
    def __init__(self, weight=0.5, D_weight=0.5):
        super(CE_DiceLoss3, self).__init__()
        self.CE_DiceLoss = CE_DiceLoss(D_weight=D_weight)
        self.weight = weight

    def forward(self, prediction, gt, prediction_s, gt_s, prediction_l, gt_l):
        return self.weight * self.CE_DiceLoss(prediction, gt) + (1 - self.weight) / 2 * self.CE_DiceLoss(
            prediction_s, gt_s) + (1 - self.weight) / 2 * self.CE_DiceLoss(prediction_l, gt_l)
