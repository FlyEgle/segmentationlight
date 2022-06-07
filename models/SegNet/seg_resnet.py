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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    def __init__(self, name, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if name.lower() == "encoder":
            self.expansion = 4
        elif name.lower() == "decoder":
            self.expansion = 1 / 4.
        
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


# build SegNetResNet50
class SegResNet50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=21, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=True):
        super(SegResNet50, self).__init__()
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
        self.layer1 = self._make_layer('encoder', block, 64, layers[0])
        self.layer2 = self._make_layer('encoder', block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer('encoder', block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer('encoder', block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.decoder_layers = [2, 2, 2, 2]

        # decoder
        self.d_layer1 = self._make_layer("decoder", block, 2048, self.decoder_layers[3], stride=1, 
                                         dilate=False)
        self.d_layer2 = self._make_layer("decoder", block, 1024, self.decoder_layers[2], stride=1, 
                                         dilate=False)
        # print(self.d_layer2)
        self.d_layer3 = self._make_layer("decoder", block, 512, self.decoder_layers[1], stride=1,
                                         dilate=False)
        self.d_layer4 = self._make_layer("decoder", block, 256, self.decoder_layers[0], stride=1)
        self.d_layer5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classification = nn.Conv2d(64, num_classes, 1, 1)

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

        # print(self.state_dict()['conv1.weight'])
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(model_urls["resnet50"], progress=True)
            new_state_dict = {}
            for k, v in self.state_dict().items():
                if k in state_dict:
                    new_state_dict[k] = state_dict[k]
                else:
                    new_state_dict[k] = v 
            self.load_state_dict(new_state_dict) 
        # print(self.state_dict()['conv1.weight'])

    def _make_layer(self, name, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if name.lower() == "encoder":
            expansion = 4
        elif name.lower() == "decoder":
            expansion =  1. / 4

        if stride != 1 or self.inplanes != int(planes * expansion):
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, int(planes * expansion), stride),
                    norm_layer(int(planes * expansion)),
                )

        layers = []
        layers.append(block(name, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = int(planes * expansion)
        for _ in range(1, blocks):
            layers.append(block(name, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # 256
        x = self.layer2(x) # 512
        x = self.layer3(x) # 1024
        x = self.layer4(x) # 2048

        return x

    def forward(self, x):
        x = self._forward_features(x)
        
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_layer1(x)
        
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_layer2(x)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_layer3(x)
        
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        x = self.d_layer4(x)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        out = self.d_layer5(x)
        x = x + out

        outputs = self.classification(x)
        return outputs


if __name__ == '__main__':
    model  = SegResNet50()
    # print(model)
    inputs = torch.randn(1,3,512,512)
    out = model(inputs)
    print(out.shape)
    # print(outputs.shape)
    