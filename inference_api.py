"""Inference 
@author: FlyEgle
@datetime: 2022-01-26
"""
import os
import cv2
import json  
import torch
import shutil 
import numpy as np 
import torch.nn.functional as F 

from models.model_factory import ModelFactory
from utils.DataAugments import Normalize, Scale, ToTensor
from torch.cuda.amp import autocast as autocast 

from tqdm import tqdm
from PIL import Image 


def aug(images):
    images, _ = Scale((800, 800))(images, images)
    images, _ = Normalize(normalize=True)(images, images)
    images, _ = ToTensor()(images, images)
    return images 


def load_ckpt(net, model_ckpt):
    state_dict = torch.load(model_ckpt, map_location="cpu")['state_dict']
    net.load_state_dict(state_dict)
    print(f"load the ckpt {model_ckpt}")
    return net 


class SegNet:
    def __init__(self, model_name, num_classes, weights):
        self.model_name = model_name 
        self.num_classes = num_classes 
        self.weights = weights 

        # build model 
        model_factory = ModelFactory()
        self.net = model_factory.getattr(model_name)(num_classes=self.num_classes)
        load_ckpt(self.net, self.weights)

        # cuda & eval 
        if torch.cuda.is_available():
            self.net.cuda()
        
        self.net.eval()

    @torch.no_grad()
    def infer(self, images):
        """images: np.ndarray RGB
        Return:
            mask : 0-1 uint8 map 
            mask_map: crop RGB images
        """
        src_h, src_w, c = images.shape 
        img = aug(images)
        img.unsqueeze_(0)  # chw->bchw

        with autocast():
            img = img.cuda()
            outputs = self.net(img)  # outputs is a list

        output = outputs[0]
        output = F.interpolate(output, size=[src_h, src_w], mode='bilinear', align_corners=True)
        # # matting
        output = torch.sigmoid(output) # b, 1, h, w
        output = output.cpu().numpy()

        # if use the ce loss trainig with softmax
        # output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        # output = output.unsqueeze(1)
        # output = output.cpu().numpy()

        mask = output 
        
        # binary threshold can control to make the context more 
        output[output >= 0.9] = 1
        output[output < 0.9] = 0

        # make uint8 mask 
        # mask = output
        mask = output.astype(np.uint8)

        # crop images
        mask_map = np.zeros((mask.shape[2], mask.shape[3], 3))
        mask_map[:,:,0] = mask[0,0,:,:] * images[:,:,0]
        mask_map[:,:,1] = mask[0,0,:,:] * images[:,:,1]
        mask_map[:,:,2] = mask[0,0,:,:] * images[:,:,2]

        mask_map_white = mask_map.copy()
        mask_map_white[mask_map_white==0] = 255

        return mask[0,0,:,:]*255,  mask_map, mask_map_white


def load_data(path):
    if os.path.isdir(path):
        data_list = [os.path.join(path, x) for x in os.listdir(path)]
    else:
        data_list = [x.strip() for x in open(path).readlines()]
        if "image_path" in data_list[0]:
            data_list = [json.loads(x)["image_path"] for x in data_list]

    return data_list 


def concat_image(image1, image2):
    w,  h = image1.size 
    new = Image.new("RGB", (w*2, h), 255)
    new.paste(image1, (0, 0))
    new.paste(image2, (w, 0))
    new_img = new.resize((w // 4, h // 4))
    return new_img



if __name__ == "__main__":
    weights = "/data/jiangmingchao/data/AICKPT/Seg/U2Net/400_epoch_800_crop_0.3_1.2_1E-4_circle_2_27k_copypaste_sgd/checkpoints/best_ckpt_epoch_133_losses_0.21228863086019242_miou_0.9878880708590272.pth"
    model_name = "u2net"
    num_classes = 1
    model = SegNet(model_name, num_classes, weights)

    # -----------------------single image inference-------------------------------------
    # test_image_path = "/data/jiangmingchao/data/code/SegmentationLight/111.png"

    # image = cv2.imread(test_image_path)
    # img = Image.fromarray(image)
    # # print(img.mode)
    # # image = image[:,:,:]

    # if image is not None:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     mask, mask_map, mask_map_w = model.infer(image)
    #     # merge = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
    #     color_mask = mask.copy()
    #     color_mask = np.concatenate(
    #         (np.expand_dims(color_mask, axis=-1),
    #         np.expand_dims(color_mask, axis=-1),
    #         np.expand_dims(color_mask, axis=-1),
    #         ), -1)
    #     # print(color_mask[color_mask==255])
    #     color_mask[:,:,0][color_mask[:,:,0]==255] = 128
    #     color_mask[:,:,1][color_mask[:,:,1]==255] = 0
    #     color_mask[:,:,2][color_mask[:,:,2]==255] = 128

    #     merge = 0.5 * image[:,:,::-1] + 0.5 * color_mask
    #     merge = merge.astype(np.uint8)
            
    #     cv2.imwrite("/data/jiangmingchao/data/code/SegmentationLight/tmp/mask.png", mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     cv2.imwrite("/data/jiangmingchao/data/code/SegmentationLight/tmp/mask_map.png", mask_map[:,:,::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     cv2.imwrite("/data/jiangmingchao/data/code/SegmentationLight/tmp/mask_map_w.png", mask_map_w[:,:,::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     cv2.imwrite("/data/jiangmingchao/data/code/SegmentationLight/tmp/merge.png", merge, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # else:
    #     raise IOError(f"{test_image_path} is not exists!!!")

    # ------------------------ folder inference -------------------------------------------
    test_folder = "/data/jiangmingchao/data/code/cluster/shein_2k_1w.log"
    # test_image_list = [os.path.join(test_folder, x) for x in os.listdir(test_folder)]
    
    out_folder = "/data/jiangmingchao/data/dataset/shein_2k_1w_720_patch/mask"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        shutil.rmtree(out_folder)
        os.makedirs(out_folder)

    test_image_list = load_data(test_folder)
    for img_path in tqdm(test_image_list):
        img = cv2.imread(img_path)
        if img is not None :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask, mask_map, _ = model.infer(img)
            img_name = img_path.split('/')[-1].split('.')[0]+'.png'
            cv2.imwrite(os.path.join(out_folder, img_name), mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            raise IOError(f"{img_path} is not exists!!!")


