"""
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
@author: FlyEgle
@datetime: 2022-01-15
"""
import torch 
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 


VGG_CKPT = "/data/jiangmingchao/data/AICKPT/r50_losses_1.0856279492378236.pth"


class ConvBnRelu(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1, padding=1, bn=True):
        super(ConvBnRelu, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding 
        self.stride = stride
        self.BN = bn

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            bias=False)
        self.relu = nn.ReLU(inplace=True)
        if self.BN:
            self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        if self.BN:
            x = self.relu(self.bn(self.conv(x)))
        else:
            x = self.relu(self.conv(x))

        return x 

class ConvBn(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1, padding=1, bn=True):
        super(ConvBn, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding 
        self.BN = bn

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )
        if self.BN:
            self.bn = nn.BatchNorm2d(self.out_channels)
    
    def forward(self, x):
        if self.BN:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)
        return x 

        
class Block(nn.Module):
    def __init__(self, layer_name, layer_num, in_channels, out_channels, block_id=1):
        super(Block, self).__init__()
        self.layer = layer_num 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_list = []

        for _ in range(self.layer):
            if layer_name.lower() == "convbnrelu":
                self.layer_list.append(ConvBnRelu(
                    3, self.in_channels, self.out_channels
                ))
            elif layer_name.lower() == "convbn":
                self.layer_list.append(ConvBn(
                    3, self.in_channels, self.out_channels
                ))

            self.in_channels = self.out_channels
        
        self.block = nn.Sequential(*self.layer_list)

    def forward(self, x):
        return self.block(x)


class SegNetVgg16(nn.Module):
    def __init__(self, num_classes):
        super(SegNetVgg16, self).__init__()
        
        self.num_classes = num_classes
        # Encoder Blocks
        self.block1 = Block("convbnrelu", 2, 3, 64)
        self.block2 = Block("convbnrelu", 2, 64, 128)
        self.block3 = Block("convbnrelu", 3, 128, 256)
        self.block4 = Block("convbnrelu", 3, 256, 512)
        self.block5 = Block("convbnrelu", 3, 512, 512)

        # Encoder Poolings
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), return_indices=True)
        
        # smooth
        # self.smoothblock = Block("convbnrelu", 3, 512, 512)

        # Decoder
        self.d_block1 = Block("convbnrelu", 3, 512, 512)
        self.d_block2 = Block("convbnrelu", 3, 512, 256)
        self.d_block3 = Block("convbnrelu", 3, 256, 128)
        self.d_block4 = Block("convbnrelu", 2, 128, 64)
        self.d_block5 = nn.Sequential(
            ConvBnRelu(3, 64, 64, 1, 1),
            ConvBnRelu(3, 64, 64, 1, 1),
            )
        
        self.classification = nn.Conv2d(64, self.num_classes, 1, 1)

        # Decoding Poolings
        self.d_pool1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.d_pool2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.d_pool3 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.d_pool4 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
        self.d_pool5 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.d_pool1 = nn.Upsample(scale_factor=2.0, mode='bilinear')
        # self.d_pool2 = nn.Upsample(scale_factor=2.0, mode='bilinear')
        # self.d_pool3 = nn.Upsample(scale_factor=2.0, mode='bilinear')
        # self.d_pool4 = nn.Upsample(scale_factor=2.0, mode='bilinear')
        # self.d_pool5 = nn.Upsample(scale_factor=2.0, mode='bilinear')

        self._initialize_weights()
        self._load_pretrain(VGG_CKPT)

    def forward(self, x):
        # encoder
        x = self.block1(x)
        x, p1_indices = self.pool1(x)
        x = self.block2(x)
        x, p2_indices = self.pool2(x)
        x = self.block3(x)
        x, p3_indices = self.pool3(x)
        x = self.block4(x)
        x, p4_indices = self.pool4(x)
        x = self.block5(x)
        x, p5_indices = self.pool5(x)

        # x = self.smoothblock(x)

        # decoder
        b, c, h, w = x.shape
        x = self.d_pool5(x, p5_indices, output_size=torch.Size([b, c, h*2, w*2]))
        # x = self.d_pool5(x)
        x = self.d_block1(x)

        b, c, h, w = x.shape
        x = self.d_pool4(x, p4_indices, output_size=torch.Size([b, c, h*2, w*2]))
        # x = self.d_pool4(x)
        x = self.d_block2(x)
        
        b, c, h, w = x.shape
        x = self.d_pool3(x, p3_indices, output_size=torch.Size([b, c, h*2, w*2]))
        # x = self.d_pool3(x)
        x = self.d_block3(x)
        
        b, c, h, w = x.shape 
        x = self.d_pool2(x, p2_indices, output_size=torch.Size([b, c, h*2, w*2]))
        # x = self.d_pool2(x)
        x = self.d_block4(x)

        b, c, h, w= x.shape
        x = self.d_pool1(x, p1_indices, output_size=torch.Size([b, c, h*2, w*2]))
        # x = self.d_pool1(x)
        x = self.d_block5(x)

        outputs = self.classification(x)
        return outputs

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def _load_pretrain(self, VGG_CKPT):
        state_dict = torch.load(VGG_CKPT, map_location="cpu")['state_dict']
        model_state_dict = self.state_dict()
        pretrain_state = {}
        for key, value in model_state_dict.items():
            if key in state_dict and value.shape == state_dict[key].shape:
                pretrain_state[key] = state_dict[key] 

        model_state_dict.update(pretrain_state)
        self.load_state_dict(model_state_dict)
        print("Load vgg imagenet pretrain!!!")


if __name__ == "__main__":
    inputs = torch.randn(1, 3, 512, 512)
    model = SegNetVgg16(21)
    print(model)
    # for k, v in model.state_dict().items():
    #     print(k)
    # print(model.state_dict().keys())
    # print(model)
    outputs = model(inputs)
    print(outputs.shape)
    



    
    





