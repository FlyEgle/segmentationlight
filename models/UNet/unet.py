""" U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Conv3x3(nn.Module):
    def __init__(self, inp, oup):
        super(Conv3x3, self).__init__()
        self.conv = nn.Sequential(
            ConvBnReLU(inp, oup, 3, 1, 1),
            ConvBnReLU(oup, oup, 3, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes=21):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        self.encoder_block1 = Conv3x3(3, 64)
        self.encoder_block2 = Conv3x3(64, 128)
        self.encoder_block3 = Conv3x3(128, 256)
        self.encoder_block4 = Conv3x3(256, 512)
        self.encoder_block5 = Conv3x3(512, 1024)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.smooth1 = ConvBnReLU(1024, 512, 1, 1, 0)
        self.smooth2 = ConvBnReLU(512, 256,  1, 1, 0)
        self.smooth3 = ConvBnReLU(256, 128, 1, 1, 0)
        self.smooth4 = ConvBnReLU(128, 64,  1, 1, 0)

        self.decoder_block1 = Conv3x3(1024, 512)
        self.decoder_block2 = Conv3x3(512, 256)
        self.decoder_block3 = Conv3x3(256, 128)
        self.decoder_block4 = Conv3x3(128, 64)

        self.classification = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):
        e1 = self.encoder_block1(x)
        
        e2 = self.pool1(self.encoder_block2(e1))     # (bs, 64, 256, 256)
        e3 = self.pool2(self.encoder_block3(e2))     # (bs, 256, 128, 128)
        e4 = self.pool3(self.encoder_block4(e3))     # (bs, 512, 64, 64)
        x =  self.pool4(self.encoder_block5(e4))      # (bs, 1024, 32, 32)
        
        x = self.smooth1(x)                          
        d5 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True) # (bs, 512, 64, 64)
        
        x = self.decoder_block1(torch.cat([e4, d5], dim=1))
        x = self.smooth2(x)
        d4 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        
        x = self.decoder_block2(torch.cat([e3, d4], dim=1))
        x = self.smooth3(x)
        d3 =  F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        
        x = self.decoder_block3(torch.cat([e2, d3], dim=1))
        x = self.smooth4(x)
        d2 = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
        
        out = self.decoder_block4(torch.cat([e1, d2], dim=1))

        output = self.classification(out)
        return output


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 512, 512)
    unet = UNet()
    outputs = unet(inputs)
    print(outputs.shape)
    
