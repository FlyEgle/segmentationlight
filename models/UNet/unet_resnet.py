"""
Unet with Resnet50 backbone 
"""
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F 


__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBnReLU(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, stride, padding),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x 



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3 = norm_layer(int(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, encoder=True):
        if encoder:
            expansion = 4
            block.expansion = expansion
        else:
            expansion = 1 / 2.
            block.expansion = expansion
            self.inplanes = planes

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes !=int(planes * expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, int(planes * expansion), stride),
                norm_layer(int(planes * expansion)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = int(planes * expansion)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.conv1(x)          # b, 64, 256, 256
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        # b, 64, 128, 128
        f1 = self.layer1(x)         # b, 256, 64, 64
        f2 = self.layer2(f1)        # b, 512, 32, 32
        f3 = self.layer3(f2)        # b, 1024, 16, 16
        f4 = self.layer4(f3)        # b, 2048, 8, 8
        return x, f1, f2, f3, f4

    def forward(self, x):
        return self._forward_features(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained, progress=True, **kwargs)

def resnet101(pretrained=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=pretrained, progress=True, **kwargs)

def resnet152(pretrained=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained=pretrained, progress=True, **kwargs)


# ResNet50 Unet
class UNET(nn.Module):
    def __init__(self, num_classes):
        super(UNET, self).__init__()
        self.num_classes = num_classes
        self.encoder = resnet50(pretrained=True)

        # change channel 
        self.smooth_5 = ConvBnReLU(2048, 1024, 1, 1, 0)
        self.smooth_4 = ConvBnReLU(1024, 512, 1, 1, 0)
        self.smooth_3 = ConvBnReLU(512, 256, 1, 1, 0)
        self.smooth_2 = ConvBnReLU(256, 64, 1, 1, 0)

        # decoder
        self.decoder1 = self.encoder._make_layer(Bottleneck, 2048, 3, 1, False, False)
        self.decoder2 = self.encoder._make_layer(Bottleneck, 1024, 6, 1, False, False)
        self.decoder3 = self.encoder._make_layer(Bottleneck, 512, 4, 1, False, False)
        self.decoder4 = self.encoder._make_layer(Bottleneck, 128, 3, 1, False, False)

        self.head = nn.Sequential(
            ConvBnReLU(64, 64, 3, 1, 1),
            ConvBnReLU(64, 64, 3, 1, 1),
        )
        self.out = nn.Conv2d(64, self.num_classes, 1, 1, 0)

    def forward(self, x):
        p1, p2, p3, p4, p5 = self.encoder(x)
        # print(p1.shape, p2.shape, p3.shape,  p4.shape, p5.shape)

        up1 = F.interpolate(p5, scale_factor=2.0,  mode='bilinear', align_corners=True) # bs, 2048, 32, 32
        s1  = self.smooth_5(up1)                                                        #  (bs, 512, 32, 32)
        c1 = torch.cat([p4, s1], dim=1)                                                 # (bs, 2048, 32, 32)
        x = self.decoder1(c1)                                                           # (bs, 1024, 32, 32)

        up2 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)  # (bs, 1024, 64, 64)
        s2  = self.smooth_4(up2)                                                       # (bs, 512, 64, 64)
        c2  = torch.cat([p3, s2],  dim=1)                                              # (bs, 1024, 64, 64)
        x   = self.decoder2(c2)                                                        # (bs, 512, 64, 64)

        up3 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)  # (bs, 512, 128, 128)
        s3 = self.smooth_3(up3)                                                        # (bs, 256, 128, 128)
        c3 = torch.cat([p2, s3], dim=1)                                                # (bs, 512, 128, 128)
        x = self.decoder3(c3)                                                          # (bs, 256, 128, 128)
        
        up4  = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True) # (bs, 256, 256, 256)
        s4 = self.smooth_2(up4)                                                        # (bs, 64, 256, 256)                                               
        p1 = F.interpolate(p1, scale_factor=2.0, mode='bilinear', align_corners=True)  # (bs, 64, 256, 256)
        c4 = torch.cat([p1, s4], dim=1)                                                # (bs, 128, 256, 256)
        x = self.decoder4(c4)                                                          # (bs, 64, 256, 256)                              

        out = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)  # (bs, 64, 512, 512)
        out = self.head(out)
        outputs = self.out(out)
        return outputs
        
if __name__ == '__main__':
    model  = UNET()
    # print(model)
    inputs = torch.randn(1,3,512,512)
    outputs = model(inputs)
    print(outputs.shape)
    