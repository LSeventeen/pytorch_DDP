import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, jaccard_score, f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def eval_metrics(target, predict):
    tp = (predict * target).sum()
    tn = ((1 - predict) * (1 - target)).sum()
    fp = ((1 - target) * predict).sum()
    fn = ((1 - predict) * target).sum()

    return tn, fp, fn, tp


def get_metrics(tn, fp, fn, tp):
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "Acc": np.round(acc, 4),
        "pre": np.round(pre, 4),
        "Sem": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "F1": np.round(f1, 4),
        "IOU": np.round(iou, 4)
    }


def get_metrics_full(tn, fp, fn, tp, target, output, output_b):
    auc = roc_auc_score(target, output)
    # precision, recall, thresholds = precision_recall_curve(target, output)
    # precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    # recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    # pra = np.trapz(precision, recall)
    # jc = jaccard_score(target, output_b)

    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "pre": np.round(pre, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "IOU": np.round(iou, 4),
        # "PRA": np.round(pra, 4),
        # "jc ": np.round(jc, 4),
    }
