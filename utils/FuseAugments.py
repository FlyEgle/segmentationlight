"""
@author: FlyEgle
@datetime: 2022-04-19
@describe: Make Complex Augments like Mixup, Cutmix, Mosaic
"""
import cv2 
import torch 
import numpy as np 
import torch.nn as nn 

from typing import Any


class MixUP:
    """Mixup for mixed two images
    Returns:
        mixed images, pairs of targets, lambda
    """
    def __init__(self, alpha=1.0, cuda=True) -> None:
        self.alpha = alpha 
        self.cuda = cuda 
        
    def __call__(self, img, tgt):
        if self.alpha:
            lam =  np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        bs = img.shape[0]
        if self.cuda:
            index = torch.randperm(bs).cuda()
        else:
            index = torch.randperm(bs)
        
        mixed_img = lam * img + (1 - lam) * img[index, :]
        tgt_a, tgt_b = tgt, tgt[index]

        return mixed_img, tgt_a, tgt_b, lam 


class Shrink:
    """Shrink the mask small 1 or 2 pixel
    """
    def __init__(self, shrink=1):
        self.shrink = shrink 

    # TODO: Fast Implementation
    def __call__(self, img, tgt):
        tgt[tgt==255] = 1
        for _ in range(self.shrink):
            contours, hierachy = cv2.findContours(
                tgt,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                tgt,
                contours,
                -1,
                (125, 125, 125),
                1
            )
            tgt[tgt==np.array([125, 125, 125])] = 0

        return img, tgt 


class CutMix:
    def __init__(self) -> None:
        pass

    def __call__(self, img, tgt):
        pass


class MixCriterion(nn.Module):
    def __init__(self, criterion):
        super(MixCriterion, self).__init__()
        self.criterion = criterion

    def forward(self, pred, tgts_a, tgts_b, lam):
        return lam * self.criterion(pred, tgts_a) + (1 - lam) * self.criterion(pred, tgts_b)


class Mosaic:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass