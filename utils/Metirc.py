"""Metric for Segmentation
- Pixel Accuracy(PA)
- Mean Pixel Accuracy(MPA)
- Mean Intersection over Union (MIoU)
- todo Frequency Weighted Intersection over Union(FWIoU)

@author: FlyEgle
@datetime: 2022-01-20
"""
import cv2 
import torch 
import numpy as np


def generate_mask(outputs, score=0.5):
    """build binary mask from outputs
    """
    # N 1 H W
    outputs = torch.sigmoid(outputs)
    # N H W
    outputs[outputs > score] = 1
    outputs[outputs <= score] = 0
    return outputs.detach().squeeze(1).cpu().numpy().astype(np.int32)

def confusion_matrix(y_true, y_pred):
    # this version is too slow to calculate the big image for segmentation 
    classes = np.max(y_true) + 1
    matrix = np.zeros((classes, classes), dtype=np.int32)
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            matrix[y_true[i], y_true[i]] += 1
        else:
            matrix[y_true[i], y_pred[i]] += 1
    
    return matrix

# TODO: 存在如果当前批次没有类别的情况
def fast_confusion_matrix(y_true, y_pred, num_classes):
    # this version is the faster than sklearn confusion matrix impelemation
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

    classes = num_classes
    numbers = classes * y_true + y_pred
    vector = np.bincount(numbers)
    matrix = vector.reshape((classes, classes))
    return matrix
    

def pixel_accuracy(confusion_matrix):
    """pixel accuracy, correct pixel / all pixel """
    pa = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return pa 


def mean_pixel_accuracy(confusion_matrix):
    """mean pixel accuracy"""
    Mpa = np.nanmean(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))
    return Mpa 


def mean_intersection_over_union(confusion_matrix):
    """Mean Intersection over Union """ 
    Union = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    Inter = np.diag(confusion_matrix)
    IoU = Inter / Union
    MIoU = np.nanmean(IoU)
    return MIoU


def frequency_weighted_intersection_over_union(confusion_matrix):
    """Intersection over Union with Classes weights
    """
    Union = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    Inter = np.diag(confusion_matrix)
    IoU = Inter / Union
    freq = np.sum(confusion_matrix, axis=1) / confusion_matrix.sum()
    FwIOU = (freq[freq>0] * IoU[freq>0]).sum()
    return FwIOU


# translate the outputs & targets tensor to numpy 
def make_outputs(outputs):
    if isinstance(outputs, np.ndarray):
        return np.argmax(outputs, axis=1)
    else:
        return torch.argmax(outputs, axis=1).cpu().numpy()

def make_targets(targets):
    if isinstance(targets, np.ndarray):
        return targets
    else:
        targets = targets.cpu().numpy()
        return targets


# batch calculate  
def calc_semantic_segmentation_confusion(pred_labels, gt_labels, num_classes):
    """batch accumulate confusion matrix
    """
    # debug the np.bincounts bug
    pred_labels[pred_labels<0] = 0
 
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = num_classes
    confusion = np.zeros((n_class, n_class), dtype=np.int32)    # (12, 12)
 
    for pred_label, gt_label in zip(pred_labels, gt_labels):
 
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
 
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same.')
 
        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:    #如果分类数大于预设的分类数目，则扩充一下。
            expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int32)
            expanded_confusion[0:n_class, 0:n_class] = confusion
 
            n_class = lb_max + 1
            confusion = expanded_confusion
 
        # Count statistics from valid pixels
        mask = gt_label >= 0
        confusion += np.bincount(n_class * gt_label[mask].astype(int) + pred_label[mask], minlength=n_class ** 2).reshape((n_class, n_class))
 
    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
 
    return confusion


class SegmentationMetric:
    """Segmentation Metric, get the confusion for all the batch results, each batch record the binnums, only record current rank if use the ddp traininig 
    """
    def __init__(self, name, classes):
        self.name = name 
        self.classes = classes 
        self.batch_num = 0
        self.confusion_matrix = 0
        self.total_losses = 0.0

    def update(self, outputs, targets, losses, bce=False, score=0.5):
        if bce:
            outputs = generate_mask(outputs, score=score)
        else:
            outputs = make_outputs(outputs)
        
        targets = make_targets(targets)

        # print(outputs.shape)
        # print(targets.shape)
        # N,H,W
        assert outputs.shape == targets.shape, "outputs shape must be same with the targets"
        
        # losses
        if isinstance(losses, torch.Tensor):
            self.batch_losses = losses.data.item()
        else:
            self.batch_losses = losses
        
        self.total_losses += self.batch_losses

        # batch confusion matrix
        self.batch_confusion_matrix = calc_semantic_segmentation_confusion(outputs, targets, self.classes)
        
        # all batch accumulate confusion matrix
        self.confusion_matrix += self.batch_confusion_matrix
        # batch losses
        self.batch_num += 1
        
    def reset(self):
        self.batch_num = 0
        self.confusion_matrix = 0 
        self.total_losses = 0.0

    @property
    def average(self):
        return self.total_losses / self.batch_num

    # batch 
    @property
    def batch_metric(self):
        # print(self.batch_confusion_matrix)
        batch_pa, batch_mpa, batch_miou, batch_fwiou = self._cal_metirc(self.batch_confusion_matrix)
        return batch_pa, batch_mpa, batch_miou, batch_fwiou

    # epoch
    @property
    def epoch_metric(self):
        epoch_pa, epoch_mpa, epoch_miou, epoch_fwiou = self._cal_metirc(self.confusion_matrix)
        return epoch_pa, epoch_mpa, epoch_miou, epoch_fwiou

    # return pa, mpa, miou
    def _cal_metirc(self, confusion_matrix):
        pa =  self._pa(confusion_matrix)
        mpa = self._mpa(confusion_matrix)
        miou = self._miou(confusion_matrix)
        fwiou = self._fwiou(confusion_matrix)
        return pa, mpa, miou, fwiou

    def _pa(self, confusion_matrix):
        pa = pixel_accuracy(confusion_matrix)
        return pa 

    def _mpa(self, confusion_matrix):
        mpa = mean_pixel_accuracy(confusion_matrix)
        return mpa  
    
    def _miou(self, confusion_matrix):
        miou = mean_intersection_over_union(confusion_matrix)
        return miou 

    def _fwiou(self, confusion_matrix):
        fwiou = frequency_weighted_intersection_over_union(confusion_matrix)
        return fwiou


if __name__ == "__main__":
    # batch_true = torch.empty(3, 4, 4, dtype=torch.long).random_(0, 21).numpy()
    # batch_pred = torch.empty(3, 4, 4, dtype=torch.long).random_(0, 21).numpy()

    batch_true = np.array([
        [[0,1,1],[1,2,2],[1,1,2]],
        [[0,1,1],[1,2,2],[1,1,2]],
        [[0,1,1],[1,2,2],[1,1,2]],
    ])


    batch_pred = np.array([
        [[0,1,0],[1,2,1],[0,1,0]],
        [[0,1,0],[1,2,1],[0,1,0]],
        [[0,1,0],[1,2,1],[0,1,0]],
    ])

    matrix = calc_semantic_segmentation_confusion(batch_pred, batch_true, num_classes=21)
    pa = pixel_accuracy(matrix)
    print(pa)
    mpa = mean_pixel_accuracy(matrix)
    print(mpa)
    miou = mean_intersection_over_union(matrix)
    print(miou)
    fwiou =  frequency_weighted_intersection_over_union(matrix)
    print(fwiou)


