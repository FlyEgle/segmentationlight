'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ['MobileNetv3']

model_url = "/data/jiangmingchao/data/code/SegmentationLight/models/FCN/pretrained/mbv3_large.pth.tar"


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(MobileNetV3, self).__init__()
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),           # 0
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),           # 1
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),           # 2
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),   # 3
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 4
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 5
            Block(3, 40, 240, 80, hswish(), None, 2),                       # 6
            Block(3, 80, 200, 80, hswish(), None, 1),                       # 7
            Block(3, 80, 184, 80, hswish(), None, 1),                       # 8   
            Block(3, 80, 184, 80, hswish(), None, 1),                       # 9
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),             # 10
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),            # 11
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),            # 12
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),            # 13
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),            # 14
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

        if self.pretrained:
            self._load_weights()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _load_weights(self):
        state_dict = {}
        state = torch.load(model_url, map_location="cpu")['state_dict']
        for s in state:
            state_dict[s[7:]] = state[s]

        self.load_state_dict(state_dict)
        print("Load the imagenet Pretrain!!!")

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        features = []
        for idx, block in enumerate(self.bneck):
            out = block(out)
            if idx == 1 or idx == 3 or idx == 6:
                features.append(out)
        
        out = self.hs2(self.bn2(self.conv2(out)))
        features.append(out)
        # out = F.adaptive_avg_pool2d(out, 1)
        # out = out.view(out.size(0), -1)
        # out = self.hs3(self.bn3(self.linear3(out)))
        # out = self.linear4(out)
        return features[0], features[1], features[2], features[3]


class FCNMobileNetv3_8S(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(FCNMobileNetv3_8S, self).__init__()
        self.backbone = MobileNetV3(pretrained=pretrained)
        
        self.NUM_CLASSES = num_classes
        
        self.smooth_conv1 = nn.Conv2d(960, 80, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(80)
        
        self.smooth_conv2 = nn.Conv2d(80, 40, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(40)

        self.smooth = nn.Conv2d(40, 40, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(40)

        self.classification = nn.Conv2d(40, self.NUM_CLASSES, 1, 1, 0)
        
    
    def forward(self, x):
        b, c, h, w = x.shape
        p1, p2, p3, p4 = self.backbone(x)
        
        p4 = self.relu1(self.bn1(self.smooth_conv1(p4)))
        out2 = F.interpolate(p4, size=(p3.shape[2], p3.shape[3]), mode='bilinear', align_corners=True)
        p3 = out2 + p3 
        out3 = self.relu2(self.bn2(self.smooth_conv2(p3)))
        out4 = F.interpolate(out3, size=(p2.shape[2], p2.shape[3]), mode='bilinear', align_corners=True)
        p2 = out4 + p2

        p2 = self.relu3(self.bn3(self.smooth(p2)))
        out = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=True)
        out = self.classification(out)
        return out


if __name__ == '__main__':
    net = FCNMobileNetv3_8S()
    inputs = torch.randn(1, 3, 512, 512)
    outputs = net(inputs)
    print(outputs.shape)