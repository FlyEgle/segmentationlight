"""Voc2017 Segmentation DataSet
@author: FlyEgle
@datetime: 2022-01-19
"""
# TODO: using LOGGER for print log
import os 
import cv2 
import json 
import random 
import numpy as np
import urllib.request as urt 

from torch.utils.data.dataset import Dataset
from utils.DataAugments import RandomHorizionFlip, RandomRotate, RandomCropScale2, RanomCopyPastePruneBG
from utils.DataAugments import Normalize, ToTensor, Compose, Scale, RandomGaussianBlur
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# TODO: need use the yaml config to control the hyparameters
# train transformers
def build_transformers(crop_size=(320, 320)):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    data_aug = [
        RandomCropScale2(scale_size=crop_size, scale=(0.3, 1.2), prob=0.5),
        RandomHorizionFlip(p=0.5),
        RanomCopyPastePruneBG(prob=0.1),           # copy paste for prune bg
        RandomRotate(degree=15, mode=0),
        RandomGaussianBlur(p=0.2),
    ]

    to_tensor = [
        Normalize(normalize=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensor(channel_first=True)
    ]

    final_aug = data_aug + to_tensor
    return Compose(final_aug)


# val transformers
def build_val_transformers(crop_size=(320, 320)):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    data_aug = [
        Scale(scale_size=crop_size)
    ]

    to_tensor = [
        Normalize(normalize=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensor(channel_first=True)
    ]

    final_aug = data_aug + to_tensor
    return Compose(final_aug)


class VocSemanticSegDataSet(Dataset):
    """Build the voc 2007 dataset for segmentation
    """
    def __init__(self, data_file, transformers=None, train_phase=True,  voc=False):
        super(VocSemanticSegDataSet, self).__init__()
        if not os.path.isfile(data_file):
            raise TypeError(f"{data_file} must be file type!!!")
        self.data_list = [json.loads(x.strip()) for x in open(data_file).readlines()]
        self.data_indices = [x for x in range(len(self.data_list))]
        self.train_phase = train_phase
        self.voc = voc 
        if self.train_phase:
            random.shuffle(self.data_list)

        if transformers is not None:
            self.data_aug = transformers
        else:
            self.data_aug = None 
        
    def _loadImages(self, line):
        img_path = line["image_path"]
        lbl_path = line["label_path"]

        if "http" not in img_path:
            image = cv2.imread(img_path)
            label = cv2.imread(lbl_path)
            # rm 255 border
            if self.voc:
                label = self._rm_border(label)
            else:
                label = self._make_lbl(label)

        # read oss data
        else:
            img_context = urt.urlopen(img_path).read()
            image = cv2.imdecode(np.asarray(bytearray(img_context), dtype='uint8'), cv2.IMREAD_COLOR)

            lbl_context = urt.urlopen(lbl_path).read()
            label = cv2.imdecode(np.asarray(bytearray(lbl_context), dtype='uint8'), cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label

    # remove the border 255 from label
    def _rm_border(self, seg):
        pe = np.where(seg==255)
        seg[pe] = 0
        return seg 

    # make value to label
    def _make_lbl(self, seg):
        pe = np.where(seg==255)
        seg[pe] = 1
        return seg 

    def __getitem__(self, index):
        for _ in range(10):
            try:
                line = self.data_list[index]
                img, lbl = self._loadImages(line)
                if self.data_aug is not None:
                    img, lbl = self.data_aug(img, lbl)
                return img, lbl
            except Exception as e:
                print(f"{self.data_list[index]} have {e} exception!!!")
                index = random.choice(self.data_indices) 
              
    def __len__(self):
        return len(self.data_list)



