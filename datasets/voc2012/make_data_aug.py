import os 
import cv2 
import json 
import numpy as np 


# seg_folder = "/data/jiangmingchao/data/dataset/voc2012/SegmentationClassAug"
# seg_list = [os.path.join(seg_folder, x) for x in os.listdir(seg_folder)]

# image_folder = "/data/jiangmingchao/data/dataset/voc2012/VOCdevkit/VOC2012/JPEGImages"
# image_list = [os.path.join(image_folder, x) for x in os.listdir(image_folder)]

# seg_dict = {x.split('/')[-1].split('.')[0]:x for x in seg_list}

# # image_dict = {}
# with open("/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/voc_aug/seg_train.log", "w") as file:
#     for image in image_list:
#         image_name = image.split('/')[-1].split('.')[0]
#         result = {}
#         if image_name in seg_dict:
#             result["image_path"] = image
#             result["label_path"] = seg_dict[image_name]

#             file.write(json.dumps(result) + '\n')

# train_file = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/voc_aug/seg_train.log"
# val_file = "/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/voc_aug/seg_val.log"

# train_list = [x.strip() for x in open(train_file).readlines()]
# val_list = [x.strip() for x in open(val_file).readlines()]


# count = 0
# for val in val_list:
#     if val in train_list:
#         count += 1

# print(count)

# image_file = "/data/jiangmingchao/data/dataset/voc2012/SegmentationClassAug/2007_000032.png"
# image = cv2.imread(image_file)
# image[np.where(image==255)] = 0
# cv2.imwrite("/data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/voc_aug/1.jpg", image)