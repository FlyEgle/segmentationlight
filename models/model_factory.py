"""Model Factory
@author: FlyEgle
@datetime: 2022-02-09
"""
import torch.nn as nn 

from types import FunctionType
from models.FCN.fcn import FCN8s, FCN16s, FCN32s
from models.FCN.fcn_resnet import FCNResNet50_8S, FCNResNet101_8S, FCNResNet152_8S
from models.FCN.fcn_mobilenetv3 import FCNMobileNetv3_8S
from models.FCN.fcn_mobilenetv2 import FCNMobilenetv2_8S
from models.SegNet.seg_vgg16 import SegNetVgg16
from models.SegNet.seg_resnet import SegResNet50
from models.SegNet.seg_mobilenetv2 import SegNetMobilenetV2
from models.UNet.unet import UNet # this version is not modify with the imagenet standard cnn 
from models.UNet.unet_resnet import UNET
from models.U2Net.u2net import U2NET, U2NET_L, U2NET_SCALE, U2NET_MUTIL_TASK # this version is used for 0-1 segmentation
from models.DeepLab.deeplab import make_deeplab
# seg-hrnet
from models.HRNet.hrnet_seg import get_seg_model
from models.HRNet.config.default import _C as config
from models.HRNet.config.default import update_config

# make hrnet
def makeHRNet(num_classes=1):
    args="models/HRNet/config/seg_hrnet_w48.yaml"
    update_config(config, args)
    # print(config)
    # config.dataset.num_classes = num_classes
    model = get_seg_model(config, use_fpn=True)
    return model 

class ModelFactory:
    def __init__(self):
        # model class
        self.__MODEL_DICT__ = {
            'fcn_8s': FCN8s,
            'fcn_16s': FCN16s,
            'fcn_32s': FCN32s,
            'fcn_8s_resnet50': FCNResNet50_8S,
            'fcn_8s_resnet101': FCNResNet101_8S,
            'fcn_8s_resnet152': FCNResNet152_8S,
            'fcn_8s_mobilenetv3': FCNMobileNetv3_8S,
            'fcn_8s_mobilenetv2': FCNMobilenetv2_8S,
            'segnet_vgg16': SegNetVgg16,
            'segnet_resnet50': SegResNet50, 
            'segnet_mobilenetv2': SegNetMobilenetV2,
            'unet_full': UNet,
            'unet_resnet50': UNET,
            'u2net': U2NET,
            'u2netl': U2NET_L,
            'u2net_mutil': U2NET_MUTIL_TASK,
            'u2net_scale': U2NET_SCALE,
            'hrnet': makeHRNet, 
        }
        # TODO：modify each function to model class
        # model function
        self.__FUNCTION_DICT__ = {
            'deeplab': make_deeplab,
            # 'u2net_full': U2NET,
            # 'u2net_lite': U2NETP
        }

    def setattr(self, name, value):
        if name in self.__MODEL_DICT__ or name in self.__FUNCTION_DICT__:
            print(f"{name} have been used in the model, please check or change a new name") 
        else:
            # function
            if isinstance(value, FunctionType):
                self.__FUNCTION_DICT__[name] = value  
            # class 
            else:
                self.__MODEL_DICT__[name] = value 

    def getattr(self, name):
        model_name = name.lower()
        if model_name in self.__MODEL_DICT__:
            return self.__MODEL_DICT__[model_name]
        elif model_name in self.__FUNCTION_DICT__:
            return self.__FUNCTION_DICT__[model_name]


if __name__ == '__main__':

    factory = ModelFactory()
    model = factory.getattr("unet")
    print(model)
    

    


