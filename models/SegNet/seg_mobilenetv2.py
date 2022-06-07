"""
SegNet with mobilenetv2 
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math

__all__ = ['mobilenetv2']

model_url = '/data/jiangmingchao/data/code/SegmentationLight/models/FCN/pretrained/mobilenetv2-c5e733a8.pth'

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., pretrained=True):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()
        if pretrained:
            self._load_weights()

    def forward(self, x):
        features = []
        for idx, feats in enumerate(self.features):
            x = feats(x)
            if idx==3 or idx == 6 or idx == 13 or idx == 17:
                features.append(x)
        # x = self.features(x)
        # x = self.conv(x)
        # features.append(x)
        return features[0], features[1], features[2], features[3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _load_weights(self):
        state_dict = torch.load(model_url, map_location="cpu")
        self.load_state_dict(state_dict)
        print("Load imagenet pretrain!!!")


class SegNetMobilenetV2(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(SegNetMobilenetV2, self).__init__()
        self.NUM_CLASSES = num_classes
        self.backbone = MobileNetV2(pretrained=pretrained)

        # d-block   320 -> 96
        self.d_block1 = nn.Sequential(
            InvertedResidual(320, 960, 1, 1),
            InvertedResidual(960, 320, 1, 1),
            InvertedResidual(320, 96, 1, 1)
        )
        self.d_block2 = nn.Sequential(
            InvertedResidual(96, 320, 1, 1),
            InvertedResidual(320, 96, 1, 1),
            InvertedResidual(96, 64, 1, 1)
        )
        self.d_block3 = nn.Sequential(
            InvertedResidual(64, 96, 1, 1),
            InvertedResidual(96, 64, 1, 1),
            InvertedResidual(64, 32, 1, 1)
        )
        self.d_block4 = nn.Sequential(
            InvertedResidual(32, 64, 1, 1),
            InvertedResidual(64, 32, 1, 1),
            InvertedResidual(32, 24, 1, 1)
        )
        self.d_block5 = nn.Sequential(
            InvertedResidual(24, 32, 1, 1),
            InvertedResidual(32, 24, 1, 1)
        )
        self.classification = nn.Conv2d(24, self.NUM_CLASSES, 1, 1, 0)
        
    def forward(self, x):
        b, c, h, w = x.shape
        _, _, _, p4 = self.backbone(x)
        
        # up + block
        x = F.interpolate(p4, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_block1(x)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_block2(x)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_block3(x)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_block4(x)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_block5(x)

        out = self.classification(x)
        return out 


if __name__ == '__main__':
    net = SegNetMobilenetV2(21)
    inputs = torch.randn(1, 3, 512, 512)
    # print(net)
    outputs = net(inputs)
    print(outputs.shape)
    