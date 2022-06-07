import torch
import torch.nn as nn
import torch.nn.functional as F


class ELU1(nn.Module):
    """make the soft relu == ELU + 1
    """
    def __init__(self, inplace=True) -> None:
        super(ELU1, self).__init__()
        self.inplace = inplace
        self.act = nn.ELU(inplace=self.inplace)

    def forward(self, x):
        return self.act(x) + 1


class REBNCONV(nn.Module):
    # relu bn convolution
    def __init__(self, in_ch=3,out_ch=3, dirate=1, elu=False):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        if not elu:
            self.relu_s1 = nn.ReLU(inplace=True)
        else:
            self.relu_s1 = ELU1(inplace=True) 
            # self.relu_s1 = nn.ELU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)


    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
       
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)
        
        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


# Coordinate Attention
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
 
        self.h = h
        self.w = w 
 
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
 
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
 
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
 
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
 
        return out


class Attn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Attn, self).__init__()
        assert channel % reduction == 0, "channel must be mutil reduction" 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RefineModule(nn.Module):
    def __init__(self, channel, out):
        super(RefineModule, self).__init__()
        self.attn = Attn(256)
        self.conv_up = nn.Conv2d(channel, 256, 3, 1, 1)
        self.conv_dn = nn.Conv2d(256, out, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        # self.relu1 = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv_up(x)
        x = self.relu1(x)
        x = self.attn(x)
        x = self.conv_dn(x)
        return x 


# easy up & down channel
class RefineModule2(nn.Module):
    def __init__(self, channel, out):
        super(RefineModule2, self).__init__()
        self.conv_up = nn.Conv2d(channel, 256, 3, 1, 1)
        self.conv_dn = nn.Conv2d(256, out, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        # self.relu1 = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv_up(x)
        x = self.relu1(x)
        x = self.conv_dn(x)
        return x 



##### U^2-Net ####
class U2NET(nn.Module):
    def __init__(self, num_classes=1, in_ch=3, pretrain=True, attn=False, add=False):
        super(U2NET, self).__init__()
        self.pretrain = pretrain 
        self.attn = attn 
        self.add  = add 

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,num_classes,3,padding=1)
        self.side2 = nn.Conv2d(64,num_classes,3,padding=1)
        self.side3 = nn.Conv2d(128,num_classes,3,padding=1)
        self.side4 = nn.Conv2d(256,num_classes,3,padding=1)
        self.side5 = nn.Conv2d(512, num_classes, 3, padding=1)
        self.side6 = nn.Conv2d(512, num_classes, 3, padding=1)

        if self.attn:
            self.outconv = RefineModule2(6 * num_classes, num_classes)
        elif self.add:
            self.outconv = nn.Conv2d(num_classes, num_classes, 1)
        else:
            self.outconv = nn.Conv2d(6 * num_classes, num_classes, 1)

        if self.pretrain:
            self._load_pretrain()

    def _load_pretrain(self):
        state_dict = torch.load("/data/jiangmingchao/data/code/U-2-Net/u2net_portrait.pth", map_location="cpu")
        if "state_dict" in state_dict:
            ckpt = state_dict['state_dict']
        elif "model" in state_dict:
            ckpt = state_dict['model']
        else:
            ckpt = state_dict

        # match key 
        model_state = self.state_dict()
        new_state = {}
        for k, v in model_state.items():
            if k in ckpt and v.shape == ckpt[k].shape:
                new_state[k] = ckpt[k]

        self.state_dict().update(new_state)
        # self.load_state_dict(ckpt)
        print("Load the pretrain")

    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        # inputs: [b, (numclass*6), h, w]
        if self.add:
            # out = (d1 + d2 + d3 + d4 + d5 + d6) / 6
            # d0 = self.outconv(out)
            d0 = torch.mean(torch.cat((d1, d2, d3, d4, d5, d6), 1), dim=1, keepdim=True)
        else:
            d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return d0, d1, d2, d3, d4, d5, d6

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        return d0, d1, d2, d3, d4, d5, d6

# U2NET-L
class U2NET_L(nn.Module):
    def __init__(self, num_classes=1, in_ch=3, dilation=4, pretrain=True, attn=False, add=False):
        super(U2NET_L, self).__init__()
        self.pretrain = pretrain 
        self.attn = attn 
        self.add  = add 
        self.dilation = dilation

        self.stage1 = RSU7(in_ch, 32*self.dilation, 64*self.dilation)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64*self.dilation, 32*self.dilation, 128*self.dilation)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128*self.dilation, 64*self.dilation, 256*self.dilation)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256*self.dilation, 128*self.dilation, 512*self.dilation)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512*self.dilation, 256*self.dilation, 512*self.dilation)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512*self.dilation, 256*self.dilation, 512*self.dilation)

        # decoder
        self.stage5d = RSU4F(1024*self.dilation,256*self.dilation,512*self.dilation)
        self.stage4d = RSU4(1024*self.dilation,128*self.dilation,256*self.dilation)
        self.stage3d = RSU5(512*self.dilation,64*self.dilation,128*self.dilation)
        self.stage2d = RSU6(256*self.dilation,32*self.dilation,64*self.dilation)
        self.stage1d = RSU7(128*self.dilation,16*self.dilation,64*self.dilation)

        self.side1 = nn.Conv2d(64*self.dilation,num_classes,3,padding=1)
        self.side2 = nn.Conv2d(64*self.dilation,num_classes,3,padding=1)
        self.side3 = nn.Conv2d(128*self.dilation,num_classes,3,padding=1)
        self.side4 = nn.Conv2d(256*self.dilation,num_classes,3,padding=1)
        self.side5 = nn.Conv2d(512*self.dilation, num_classes, 3, padding=1)
        self.side6 = nn.Conv2d(512*self.dilation, num_classes, 3, padding=1)

        if self.attn:
            self.outconv = RefineModule2(6 * num_classes, num_classes)
        elif self.add:
            self.outconv = nn.Conv2d(num_classes, num_classes, 1)
        else:
            self.outconv = nn.Conv2d(6 * num_classes, num_classes, 1)

        if self.pretrain:
            self._load_pretrain()

    def _load_pretrain(self):
        state_dict = torch.load("/data/jiangmingchao/data/code/U-2-Net/u2net_portrait.pth", map_location="cpu")
        if "state_dict" in state_dict:
            ckpt = state_dict['state_dict']
        elif "model" in state_dict:
            ckpt = state_dict['model']
        else:
            ckpt = state_dict

        # match key 
        model_state = self.state_dict()
        new_state = {}
        for k, v in model_state.items():
            if k in ckpt and v.shape == ckpt[k].shape:
                new_state[k] = ckpt[k]

        self.state_dict().update(new_state)
        # self.load_state_dict(ckpt)
        print("Load the pretrain")

    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        # inputs: [b, (numclass*6), h, w]
        if self.add:
            # out = (d1 + d2 + d3 + d4 + d5 + d6) / 6
            # d0 = self.outconv(out)
            d0 = torch.mean(torch.cat((d1, d2, d3, d4, d5, d6), 1), dim=1, keepdim=True)
        else:
            d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return d0, d1, d2, d3, d4, d5, d6



class SegTask(nn.Module):
    def __init__(self) -> None:
        super(SegTask, self).__init__()
        self.conv1 = REBNCONV(4, 32)
        self.conv2 = REBNCONV(32, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, x, img):
        inp = torch.cat([x, img], dim=1)
        out = self.conv3(self.conv2(self.conv1(inp)))
        return out 
        

##### U^2-Net ####
class U2NET_MUTIL_TASK(nn.Module):
    def __init__(self, num_classes=1, in_ch=3, pretrain=True, attn=False, add=False):
        super(U2NET_MUTIL_TASK, self).__init__()
        self.pretrain = pretrain 
        self.attn = attn 
        self.add  = add 

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,num_classes,3,padding=1)
        self.side2 = nn.Conv2d(64,num_classes,3,padding=1)
        self.side3 = nn.Conv2d(128,num_classes,3,padding=1)
        self.side4 = nn.Conv2d(256,num_classes,3,padding=1)
        self.side5 = nn.Conv2d(512, num_classes, 3, padding=1)
        self.side6 = nn.Conv2d(512, num_classes, 3, padding=1)

        if self.attn:
            self.outconv = RefineModule2(6 * num_classes, num_classes)
        elif self.add:
            self.outconv = nn.Conv2d(num_classes, num_classes, 1)
        else:
            # add mutil task
            self.aux = SegTask()
            self.outconv = nn.Conv2d(6 * num_classes, num_classes, 1)

        if self.pretrain:
            self._load_pretrain()

    def _load_pretrain(self):
        state_dict = torch.load("/data/jiangmingchao/data/code/U-2-Net/u2net_portrait.pth", map_location="cpu")
        if "state_dict" in state_dict:
            ckpt = state_dict['state_dict']
        elif "model" in state_dict:
            ckpt = state_dict['model']
        else:
            ckpt = state_dict

        # match key 
        model_state = self.state_dict()
        new_state = {}
        for k, v in model_state.items():
            if k in ckpt and v.shape == ckpt[k].shape:
                new_state[k] = ckpt[k]

        self.state_dict().update(new_state)
        # self.load_state_dict(ckpt)
        print("Load the pretrain")

    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        # inputs: [b, (numclass*6), h, w]
        if self.add:
            # out = (d1 + d2 + d3 + d4 + d5 + d6) / 6
            # d0 = self.outconv(out)
            d0 = torch.mean(torch.cat((d1, d2, d3, d4, d5, d6), 1), dim=1, keepdim=True)
        else:
            d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
            # sup seg task
            seg = self.aux(d0, x)

        return d0, d1, d2, d3, d4, d5, d6, seg


class ConvPool(nn.Module):
    def __init__(self, in_channels):
        super(ConvPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

##### U^2-Net ####
"""
Add Se-Attention and repalce the maxpool to conv+stride
"""
class U2NET_SCALE(nn.Module):
    def __init__(self, num_classes=1, in_ch=3, pretrain=True, attn=False, add=False, size=(320, 320)):
        super(U2NET_SCALE, self).__init__()
        self.pretrain = pretrain 
        self.attn = attn 
        self.add  = add 
        self.size = size 

        self.stage1 = RSU7(in_ch, 32, 64)
        self.attn1 = CA_Block(64, h=self.size[0], w=self.size[1])
        self.pool12 = ConvPool(64)  # self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.attn2  = CA_Block(128, h=self.size[0]//2, w=self.size[1]//2)
        self.pool23 = ConvPool(128) # self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.attn3  = CA_Block(256, h=self.size[0]//4, w=self.size[1]//4)
        self.pool34 = ConvPool(256)  # self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.attn4  = CA_Block(512, h=self.size[0]//8, w=self.size[1]//8)
        self.pool45 = ConvPool(512) # self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.attn5  = CA_Block(512, h=self.size[0]//16, w=self.size[1]//16)
        self.pool56 = ConvPool(512) # self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,num_classes,3, padding=1)
        self.side2 = nn.Conv2d(64,num_classes,3, padding=1)
        self.side3 = nn.Conv2d(128,num_classes,3, padding=1)
        self.side4 = nn.Conv2d(256,num_classes,3, padding=1)
        self.side5 = nn.Conv2d(512, num_classes,3,padding=1)
        self.side6 = nn.Conv2d(512, num_classes,3,padding=1)

        if self.attn:
            self.outconv = RefineModule2(6 * num_classes, num_classes)
        elif self.add:
            self.outconv = nn.Conv2d(num_classes, num_classes, 1)
        else:
            self.outconv = nn.Conv2d(6 * num_classes, num_classes, 1)

        if self.pretrain:
            self._load_pretrain()

    def _load_pretrain(self):
        state_dict = torch.load("/data/jiangmingchao/data/code/U-2-Net/u2net_portrait.pth", map_location="cpu")
        if "state_dict" in state_dict:
            ckpt = state_dict['state_dict']
        elif "model" in state_dict:
            ckpt = state_dict['model']
        else:
            ckpt = state_dict

        # match key 
        model_state = self.state_dict()
        new_state = {}
        for k, v in model_state.items():
            if k in ckpt and v.shape == ckpt[k].shape:
                new_state[k] = ckpt[k]

        self.state_dict().update(new_state)
        # self.load_state_dict(ckpt)
        print("Load the pretrain")

    def forward(self, x):

        hx = x
        #stage 1
        hx1 = self.stage1(hx)
        hx1 = self.attn1(hx1)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx2 = self.attn2(hx2)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx3 = self.attn3(hx3)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx4 = self.attn4(hx4)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx5 = self.attn5(hx5)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)

        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        # inputs: [b, (numclass*6), h, w]
        if self.add:
            # out = (d1 + d2 + d3 + d4 + d5 + d6) / 6
            # d0 = self.outconv(out)
            d0 = torch.mean(torch.cat((d1, d2, d3, d4, d5, d6), 1), dim=1, keepdim=True)
        else:
            d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return d0, d1, d2, d3, d4, d5, d6


if __name__ == '__main__':
    # print(U2NET)
    model = U2NET_L()
    # model = U2NET_SCALE()
    print(model)
    # state_dict = torch.load("/data/jiangmingchao/data/code/U-2-Net/u2net_portrait.pth", map_location="cpu")
    # if "state_dict" in state_dict:
    #     ckpt = state_dict['state_dict']
    # elif "model" in state_dict:
    #     ckpt = state_dict['model']
    # else:
    #     ckpt = state_dict

    # model.load_state_dict(ckpt)
    # print("load ckpt")
    # print(model)

    rand_inputs = torch.randn(1, 3, 320, 320)
    outputs = model(rand_inputs)
    print(outputs[-1].shape)
    # print(isinstance(outputs, tuple))

    # x = torch.randn(1, 16, 320, 320)
    # ca = CA_Block(channel=16, h=320, w=320)
    # out = ca(x)
    # print(out.shape)