"""Make PascalVoc2021 segmentation data log
@author: FlyEgle
@datetime: 2021-01-18
"""
import os
from turtle import pos 
import cv2 
import json 
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm 


Folder = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/Annotations/"
folder_list = [os.path.join(Folder, x) for x in os.listdir(Folder)]


def parser_annoations(data_annoations):
    tree = ET.parse(data_annoations)
    root = tree.getroot()
    
    attrDict = {}

    for child in root:
        if child.tag == "folder" or child.tag == "filename" or child.tag  == "segmented":
            attrDict[child.tag] = child.text
    
    attrDict['obj'] = []
   
    for obj in root.findall('object'):
        obj_dict = {}
        for child in obj:
            if child.tag == "truncated" or child.tag == "difficult":
                if child.tag == 1:
                    break

            if child.tag == "name":
                obj_dict["name"] = child.text 

            if child.tag == "bndbox":
                bbox = [
                    int(float(child[0].text)),
                    int(float(child[1].text)),
                    int(float(child[2].text)),
                    int(float(child[3].text)),
                ]
                obj_dict['bbox'] = bbox
            
        attrDict['obj'].append(obj_dict)
        
    return attrDict


# filter data for segmentation class or object
def filter_data(data_file):
    data_list = [json.loads(x.strip()) for x in open(data_file).readlines()]
    with open("/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data_seg.log", "w") as file:
        for data in data_list:
            if data['segmented'] == "1" and len(data["obj"]) >= 1:
                file.write(json.dumps(data) +  '\n')
        

def match_labels(data_file):
    image_folder = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/JPEGImages"
    label_folder = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/SegmentationClass"
    data_list = [json.loads(x.strip()) for x in open(data_file).readlines()]

    labels = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    ]

    labels_dict = {labels[i]:i for i in range(len(labels))}

    with open("/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data_seg_label.log", "w") as file:
        for data in data_list:
            data_dict = {}
        
            images_name = data["filename"]
            labels_name = images_name.split('/')[-1].split('.')[0] + '.png'
            images_path = os.path.join(image_folder, images_name)
            labels_path = os.path.join(label_folder, labels_name)
            
            data_dict['image_path'] = images_path
            data_dict["label_path"] = labels_path
            data_dict["obj"] = []


            for obj in data["obj"]:
                class_number = labels_dict[obj['name']]
                class_bbox = obj['bbox']
                cls_bbx_dict = {
                    "class": class_number, 
                    "bbox": class_bbox
                }
                data_dict["obj"].append(cls_bbx_dict)

            file.write(json.dumps(data_dict)  +  '\n')
        

def draw_bbox(data_dict):
    label_path =  data_dict["label_path"]
    label_image = cv2.imread(label_path)
    image_name = label_path.split('/')[-1]
    output_folder = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/out_label"
    
    for data in data_dict['obj']:
        cls = data['class']
        box = data['bbox']
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
        cv2.rectangle(label_image, (x1, y1), (x2, y2), (255,0,0), 2)

    cv2.imwrite(os.path.join(output_folder, image_name), label_image)


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

def find_contours(images, rgb_value):
    # images = cv2.imread(label_path)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    location = np.all((images == rgb_value), axis=2)
    position = np.argwhere(location==1).tolist()
    new_position = np.array([[x[1],x[0]] for x in position], dtype=np.int32)
    
    # print(position)
    return new_position


# def make_targets(images, rgb_value, targets):
#     contours = find_contours(images, rgb_value)
#     targets = cv2.fillPoly(targets, [contours], )


if __name__ == '__main__':
    # with open("/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data.log", "w") as file:
    #     for data in tqdm(folder_list):
    #         attr = filter_annoations(data)
    #         file.write(json.dumps(attr) + '\n')

    # data_file = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data.log"
    # filter_data(data_file) 
    # pass     
    # data_file = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data_seg.log"
    # match_labels(data_file)
    
    # data_file = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data_seg_label.log"
    # data_list = [json.loads(x.strip()) for x in open(data_file).readlines()]

    # data_list  = [{"image_path": "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/JPEGImages/2011_001621.jpg", "label_path": "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/SegmentationClass/2011_001621.png", "obj": [{"class": 8, "bbox": [154, 80, 296, 213]}, {"class": 19, "bbox": [500, 421, 199, 1]}, {"class": 14, "bbox": [127, 1, 362, 155]}, {"class": 14, "bbox": [216, 191, 264, 117]}]}]

    # OUTPUT_FOLDER = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/SegmentationClassTargets"
    
    # # try:
    # for data in tqdm(data_list):
    #     # draw_bbox(data)
    #     label_path =  data["label_path"]
    #     label_name = label_path.split('/')[-1]

    #     images = cv2.imread(label_path)
    #     targets = np.zeros((images.shape[0], images.shape[1]))
    #     # print(label_path)
    #     if label_name == "2011_001621.png":
    #         for obj in data['obj']:
    #             cls = int(obj['class'])
    #             if cls == 19:
    #                 color = _CLASS_LABEL_DICT[14]
    #             else:
    #                 color = _CLASS_LABEL_DICT[cls]
    #             # print(cls, color)
    #             pts = find_contours(images, color)
    #             # print(pts)
    #             # bg is 0
    #             cv2.fillPoly(targets, [pts], cls+1)
    #             # break 
            
    #         cv2.imwrite(os.path.join(os.path.join(OUTPUT_FOLDER, label_name)), targets)
    #     else:
    #         for obj in data['obj']:
    #             cls = int(obj['class'])
    #             color = _CLASS_LABEL_DICT[cls]
    #             pts = find_contours(images, color)
    #             # bg is 0
    #             cv2.fillPoly(targets, [pts], cls+1)
    #             # break 
            
    #         cv2.imwrite(os.path.join(OUTPUT_FOLDER, label_name), targets)

            # /data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/SegmentationClass/2011_001621.png

    # image_path = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/SegmentationClassTargets/2007_000032.png"
    # image = cv2.imread(image_path)

    
    # new_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    # for i in range(1, 21):
    #     location = np.all((image==i), axis=2)
    #     position = np.where(location==1)
    #     # print(position)
    #     if position[0].size > 0:
    #         new_image[position] = 255

    # cv2.imwrite("/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/2.jpg", new_image)
    # pass 

    TRAIN_IMAGE_LIST  = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
    VAL_IMAGE_LIST  = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

    output_file = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/segclass"
    label_folder = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/SegmentationClassTargets"

    data_path = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data_seg_label.log"
    data_list = [json.loads(x.strip()) for x in open(data_path).readlines()]

    train_list = {x.strip():0 for x in open(TRAIN_IMAGE_LIST).readlines()}
    val_list = {x.strip():0 for x in open(VAL_IMAGE_LIST).readlines()}


    train_log = open(os.path.join(output_file, "seg_train.log"), "w")
    val_log = open(os.path.join(output_file, "seg_val.log"), "w")

    for data in data_list:
        image_path = data["image_path"]
        image_name = image_path.split('/')[-1].split('.')[0]
        label_path = os.path.join(label_folder, image_name+'.png')
        result = {
            "image_path": image_path,
            "label_path": label_path
        }
        
        if image_name in train_list:
            train_log.write(json.dumps(result) + '\n')
        elif image_name in val_list:
            val_log.write(json.dumps(result) + '\n')


    
