"""Batch inference
@author:Flyegle
@datetime: 2022-06-02
"""
import os 
import cv2 
import json 
import torch 
import random
import numpy as np 
import torch.nn.functional as F 

from models.model_factory import ModelFactory
from utils.DataAugments import Normalize, Scale, ToTensor
from torch.cuda.amp import autocast as autocast
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
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


class DataSet(Dataset):
    def __init__(self, file):
        super(DataSet, self).__init__()
        self.data_list = []
        self._load_data(file)
        self.data_index = [x for x in range(len(self.data_list))]

    def _load_data(self, file):
        if os.path.isdir(file):
            self.data_list = [os.path.join(file, x) for x in os.listdir(file)]
        elif os.path.isfile(file):
            data_list = [x.strip() for x in open(file).readlines()]
            if "image_path" in data_list[0]:
                self.data_list = [json.loads(x)["image_path"] for x in data_list]
            else:
                self.data_list = data_list 
        else:
            raise IOError(f"{file} must be images folder or meta log")

    def __getitem__(self, index):
        for _ in range(10):
            try:
                image = cv2.imread(self.data_list[index])
                h, w, _ = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = aug(image)
                image_name = self.data_list[index].split('/')[-1].split('.')[0]
                return image_tensor, (w, h),  image_name 
            except Exception as e:
                print(f"{self.data_list[index]} have some error, need change another!!! {e}")
                index = random.choice(self.data_index)
                
    def __len__(self):
        return len(self.data_list)


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
            outputs: torch.Tensor
        """
        with autocast():
            images = images.cuda()
            outputs = self.net(images)
        return outputs 


def post_process(outputs, shape, name, out_folder):
    outputs = outputs[0]
    b, c, h, w = outputs.shape 
    for i in range(b):
        output = F.interpolate(outputs[i,:,:,:].unsqueeze(0), size=[shape[1][i], shape[0][i]], mode="bilinear", align_corners=True)
        output = torch.sigmoid(output)
        output = output.permute(0,2,3,1).cpu().numpy()
        # b, c, h, w
        output[output >= 0.9] = 1
        output[output < 0.9] = 0

        output = output * 255
        mask = output.astype(np.uint8)
        output = np.concatenate((mask[0,:,:,:], mask[0,:,:,:],mask[0,:,:,:]), axis=-1)
        
        path = os.path.join(out_folder, name[i]+'.png')
        cv2.imwrite(path, output)



if __name__ == '__main__':

    model = SegNet(model_name="u2net", num_classes=1, weights="/data/jiangmingchao/data/AICKPT/Seg/U2Net/400_epoch_800_crop_0.3_1.2_1E-4_circle_2_27k_copypaste_sgd/checkpoints/best_ckpt_epoch_133_losses_0.21228863086019242_miou_0.9878880708590272.pth")
    
    file = "/data/jiangmingchao/data/code/cluster/shein_2k_1w.log"
    out_file = "/data/jiangmingchao/data/dataset/shein_2k_1w_720_patch/mask_batch"

    if not os.path.exists(out_file):
        os.makedirs(out_file)

    dataset = DataSet(file)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=None,
        num_workers=32
    )

    length = len(loader)
    for idx, data in enumerate(loader):
        images, shape, name = data
        outputs = model.infer(images)

        post_process(outputs, shape, name, out_file)
        print(f"process {idx}/{length} mask!!!")


            
