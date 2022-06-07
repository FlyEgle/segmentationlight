import os 
import cv2 
import json 
import torch 
import random 
from tqdm import tqdm 
# from torch.utils.data import Dataset, DataLoader
# from dataset import build_val_transformers, VocSemanticSegDataSet
# data_file = "/data/jiangmingchao/data/code/U-2-Net/makeData/taobao_seg_no_repeat_train.log"
# data_list = [x.strip() for x in open(data_file).readlines()]


# for data in tqdm(data_list):
#     data_json = json.loads(data)


# data_file = "/data/jiangmingchao/data/code/U-2-Net/makeData/taobao_seg_no_repeat_val.log"
# data_file = "/data/jiangmingchao/data/code/U-2-Net/makeData/taobao_green_shein/shin_10k.log"
# data_list = open()
# dataset = VocSemanticSegDataSet(
#     data_file, build_val_transformers(crop_size=(800, 800)), train_phase=False
# )

# dataloader = DataLoader(
#     dataset,
#     batch_size=10,
#     shuffle=True,
#     num_workers=32,
#     pin_memory=True
# )

# for idx, batch in enumerate(dataloader):
#     print(idx, batch.shape)
# new_folder = "/data/jiangmingchao/data/dataset/shein/images_copy"
# data_list = [json.loads(x.strip())["image_path"] for x in open(data_file).readlines()][10647:10648]
# for data in tqdm(data_list):
#     print(data)
#     image_name = data.split('/')[-1]
#     image = cv2.imread(data)
    # cv2.imwrite(os.path.join(new_folder, image_name), image)


data_file = "/data/jiangmingchao/data/code/U-2-Net/makeData/taobao_green_shein/shein_no_use_10k.log"
data_list = [x.strip() for x in open(data_file).readlines()]


sample_list = random.sample(data_list, 1000)

with open("/data/jiangmingchao/data/code/U-2-Net/makeData/taobao_green_shein/shein_sample_no_use_1k.log", "w") as file:
    for data in sample_list:
        file.write(data + '\n')