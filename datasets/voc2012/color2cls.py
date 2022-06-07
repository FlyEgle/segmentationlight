"""Translate the color map to class
@author: FlyEgle
@datetime: 2022-01-19
"""
import os 
import cv2  
import json 
import numpy as np 
from tqdm import tqdm 

# bg is not in class 
_CLASS_LABEL_DICT = {
    0: (128, 0, 0),
    1: (0, 128, 0),
    2: (128, 128, 0),
    3: (0, 0, 128),
    4: (128, 0, 128),
    5: (0, 128, 128),
    6: (128, 128, 128),
    7: (64, 0, 0),
    8: (192, 0, 0),
    9: (64, 128, 0),
    10: (192, 128, 0),
    11: (64, 0, 128),
    12: (192, 0, 128),
    13: (64, 128, 128),
    14: (192, 128, 128),
    15: (0,  64, 0),
    16: (128, 64, 0),
    17: (0, 192, 0),
    18: (128, 192, 0),
    19: (0,64,128)
}

border = (224, 224, 192)


def find_contours(images, rgb_value):
    location = np.all((images == rgb_value), axis=2)
    position = np.where(location==1)

    return position


def remove_border(images, border=border):
    location = np.all((images==border), axis=2)
    position = np.where(location==1)
    images[position] = 0
    return images


if __name__ == '__main__':
    data_file = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data_seg_label.log"
    data_list = [json.loads(x.strip()) for x in open(data_file).readlines()]

    OUTPUT_FOLDER = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/SegmentationClassTargets"

    for data in tqdm(data_list):
        label_path =  data["label_path"]
        label_name = label_path.split('/')[-1]
        images = cv2.imread(label_path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        targets = np.zeros((images.shape[0], images.shape[1]))

        if label_name == "2011_001621.png":
            for obj in data['obj']:
                cls = int(obj['class'])
                if cls == 19:
                    color = _CLASS_LABEL_DICT[14]
                else:
                    color = _CLASS_LABEL_DICT[cls]

                pts = find_contours(images, color)
                # bg is 0
                targets[pts] = cls+1
            
            cv2.imwrite(os.path.join(os.path.join(OUTPUT_FOLDER, label_name)), targets)
        else:
            for obj in data['obj']:
                cls = int(obj['class'])
                color = _CLASS_LABEL_DICT[cls]
                pts = find_contours(images, color)
                # bg is 0
                targets[pts] = cls+1
                # break 
            
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, label_name), targets)