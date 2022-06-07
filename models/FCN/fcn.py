"""
Fully Convolutional Networks for Semantic Segmentation
@author: FlyEgle
@datetime: 2022-01-15
"""
import torch 
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F 


VGG_CKPT = "/data/jiangmingchao/data/AICKPT/r50_losses_1.0856279492378236.pth"


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


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
            stride=self.stride)
        self.relu = nn.ReLU(inplace=True)
        if self.BN:
            self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        if self.BN:
            x = self.relu(self.bn(self.conv(x)))
        else:
            x = self.relu(self.conv(x))

        return x 

class Block(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, block_id=1):
        super(Block, self).__init__()
        self.layer = layer_num 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_list = []

        for _ in range(self.layer):
            self.layer_list.append(ConvBnRelu(
                3, self.in_channels, self.out_channels
            ))
            self.in_channels = self.out_channels
        
        self.block = nn.Sequential(*self.layer_list)

    def forward(self, x):
        return self.block(x)


class FCN32s(nn.Module):
    """Based on vgg16, only have 32 strides for outputs"""
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        self.num_classes = num_classes
        # block
        self.block1 = Block(2, 3, 64, block_id=1)
        self.block2 = Block(2, 64, 128, block_id=2)
        self.block3 = Block(3, 128, 256, block_id=3)
        self.block4 = Block(3, 256, 512, block_id=4)
        self.block5 = Block(3, 512, 512, block_id=5)

        # pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        
        # head 
        self.fc6 = ConvBnRelu(3, 512, 4096, 1, 0)
        # self.drop6 = nn.Dropout(p=0.1, inplace=True)
        self.fc7 = ConvBnRelu(1, 4096, 4096, 1, 0)
        # self.drop7 = nn.Dropout(p=0.1, inplace=True)
        # self.proj = ConvBnRelu(3, 512, 512, 1, 0)
        self.score_fr = nn.Conv2d(4096, self.num_classes, 3)

        self._initialize_weights()
        self._load_pretrain(VGG_CKPT)

    def forward(self, x):
        b, c, h, w = x.shape
        # encoder
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.block5(x))

        # head
        # x = self.drop6(self.fc6(x))
        # x = self.drop7(self.fc7(x))
        x = self.fc7(self.fc6(x))
        x = self.score_fr(x)

        # prediction
        outputs = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        return outputs

    def _initialize_weights(self):
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


class FCN16s(nn.Module):
    """Based on vgg16, have 32 strides & 16 strides for outputs"""
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()
        self.num_classes = num_classes
        
        # block
        self.block1 = Block(2, 3, 64)
        self.block2 = Block(2, 64, 128)
        self.block3 = Block(3, 128, 256)
        self.block4 = Block(3, 256, 512)
        self.block5 = Block(3, 512, 512)

        # pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # head 
        self.conv6 = ConvBnRelu(3, 512, 4096, 1, 1)
        self.conv7 = ConvBnRelu(3, 4096, 4096, 1, 1)
        self.classification = nn.Conv2d(4096, self.num_classes, 1, 1)

        # intermidate layer
        self.conv_output16 = ConvBnRelu(3, 512, 4096, 1, 1)

        self._initialize_weights()
        self._load_pretrain(VGG_CKPT)

    def forward(self, x):
        b, c, h, w = x.shape
        # encoder
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        inp1 = x 
        # output 16s
        out1 = self.conv_output16(inp1)  # (b, num_classes, input_size/16, input_size/16)
        x = self.pool5(self.block5(x))

        # head
        x = self.conv6(x)
        x = self.conv7(x)

        # pool4 + outputs 
        x = F.interpolate(x, size=(x.shape[2]*2, x.shape[3]*2), mode='bilinear', align_corners=True)
        x = x + out1 
        x = self.classification(x)
        # prediction
        outputs = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
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

    
class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        # block
        self.block1 = Block(2, 3, 64)
        self.block2 = Block(2, 64, 128)
        self.block3 = Block(3, 128, 256)
        self.block4 = Block(3, 256, 512)
        self.block5 = Block(3, 512, 512)
        # pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # head 
        self.conv6 = ConvBnRelu(3, 512, 4096, 1, 1)
        self.conv7 = ConvBnRelu(3, 4096, 4096, 1, 1)
        self.classification = nn.Conv2d(4096, self.num_classes, 1, 1)

        # intermidate layer
        self.conv_output16 = ConvBnRelu(3, 512, 4096, 1, 1)
        self.conv_output8 = ConvBnRelu(3, 256, 4096, 1, 1)

        self._initialize_weights()
        self._load_pretrain(VGG_CKPT)

    def forward(self, x):
        b, c, h, w = x.shape
        # encoder
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        
        # output 8s
        inp1 = x
        out1 = self.conv_output8(inp1)
        x = self.pool4(self.block4(x))
        
        # output 16s
        inp2 = x 
        out2 = self.conv_output16(inp2)  # (b, num_classes, input_size/16, input_size/16)
        x = self.pool5(self.block5(x))

        # head
        x = self.conv6(x)
        x = self.conv7(x)

        # pool4 + outputs x 2  16 x
        x = F.interpolate(x, size=(x.shape[2]*2, x.shape[3]*2), mode='bilinear', align_corners=True) 
        x = x + out2

        # pool3 + outputs x2   8 x
        x = F.interpolate(x, size=(x.shape[2]*2, x.shape[3]*2), mode='bilinear', align_corners=True)
        x = x + out1 

        x = self.classification(x)
        # prediction
        outputs = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
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

def load_pretrain(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")['state_dict']
    model_state_dict = model.state_dict()
    pretrain_state = {}
    for key, value in model_state_dict.items():
        if key in state_dict and value.shape == state_dict[key].shape:
            pretrain_state[key] = state_dict[key] 
    
        # if key == "fc6" or key == "fc7":
        #     pretrain_state[key] = state_dict[key].view(pretrain_state[key].size())

    model_state_dict.update(pretrain_state)
    model.load_state_dict(model_state_dict)
    print("Load vgg imagenet pretrain!!!")
    return model 


if __name__ == "__main__":
    inputs = torch.randn(1, 3, 300, 500)
    model = FCN32s(21)
    print(model)
    # for k, v in model.state_dict().items():
    #     print(k)
    # print(model.state_dict().keys())
    # print(model)
    outputs = model(inputs)
    print(outputs.shape)
    



    
    

