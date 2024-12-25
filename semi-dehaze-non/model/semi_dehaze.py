import torch
import torch.nn as nn
import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F
from resnext.resnext101 import ResNeXt101
from model.common import UNet
import numpy as np
from model import common
from model.common import UNet
import math
import matplotlib.pyplot as plt
from torchstat import stat
def make_model():
    return dehaze()

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class SWRCAB(nn.Module):
    def __init__(self, inlayer=64, outlayer=64):
        super(SWRCAB, self).__init__()

        self.inlayer = inlayer
        self.outlayer = outlayer

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.attention_block = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=1, padding=1)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outlayer, outlayer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        x = self.conv_block1(x)
        attention = self.attention_block(x)
        x = self.conv_block2(x)
        weights = x * attention

        weights = self.gap(weights).view(-1, self.outlayer)
        weights = self.fc(weights)
        weights = self.sigmoid(weights)

        weights = weights.view(-1, self.outlayer, 1, 1)

        x = x * weights

        x = x + residual

        return x


class ResidualInResiduals(nn.Module):

    def __init__(self, inner_channels=32, block_count=10):
        super(ResidualInResiduals, self).__init__()

        self.res_blocks = nn.ModuleList([SWRCAB(inner_channels, inner_channels) for i in range(block_count)])
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        residual = x

        for i, _ in enumerate(self.res_blocks):
            x = self.res_blocks[i](x)

        x = self.conv_block1(x)
        x = x + residual

        return x



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Mix(nn.Module):
    def __init__(self, m=1):
        super(Mix, self).__init__()
        w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2, feat3):
        factor = self.mix_block(self.w)
        other = (1 - factor)/2
        output = fea1 * other.expand_as(fea1) + fea2 * factor.expand_as(fea2) + feat3 * other.expand_as(feat3)
        return output, factor

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()

        assert kernel_size % 2== 1
        self.padding = (kernel_size -1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return  F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)

class SmoothDilated(nn.Module):
    def __init__(self, dilation=1, group=1):
        super(SmoothDilated, self).__init__()

        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        out1 = self.pre_conv1(x)
        out2 = self.conv1(out1)
        out3 = self.norm1(out2)

        return F.relu(x+out3)

class DCCL(nn.Module):
    def __init__(self):
        super(DCCL, self).__init__()

        self.conv1 = SmoothDilated(dilation=1)
        self.conv2 = SmoothDilated(dilation=3)
        self.conv3 = SmoothDilated(dilation=5)

        self.conv = nn.Conv2d(192, 64, 1, 1, 0, bias=False)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out = torch.cat([out1, out2, out3], 1)

        final_out = self.conv(out)

        return final_out

class residual_block(nn.Module):
    def __init__(self):
        super(residual_block, self).__init__()

        self.DCCL_block1 = DCCL()
        self.BN1 = nn.BatchNorm2d(64)
        self.PReLU1 = nn.PReLU(num_parameters=1, init=0.25)


    def forward(self, x):
        out1 = self.DCCL_block1(x)
        out2 = self.BN1(out1)
        out3 = self.PReLU1(out2)

        out = torch.add(out3, x)

        return out


class Detail_Net(nn.Module):
    def __init__(self):
        super(Detail_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.PReLU1 = nn.PReLU(num_parameters=1, init=0.25)

        self.residual = residual_block()

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1,bias=False)
        self.BN1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.PReLU1(out1)
        out = out2
        out3 = self.residual(out2)
        out4 = self.conv2(out3)
        out5 = self.BN1(out4)
        out6 = torch.add(out5, out)
        out7 = self.conv3(out6)

        return out7


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter*(2**i), num_filter*(2**(i+1)), kernel_size, stride, padding, bias, activation, norm=None)
            )

            self.up_convs.append(
                DeconvBlock(num_filter*(2**(i+1)), num_filter*(2**i), kernel_size, stride, padding, bias, activation, norm=None)
            )


    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft- len(ft_l_list) + i](ft_h)
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft-i-1](ft_fusion - ft_l_list[i]) + ft_h_list[len(ft_l_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i+1):
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion

class Encoder_MDCBlock1(torch.nn.Module):

    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None, mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter//(2**i), num_filter//(2**(i+1)), kernel_size, stride, padding, bias, activation, norm=None)
            )
            self.down_convs.append(
                ConvBlock(num_filter//(2**(i+1)), num_filter//(2**i), kernel_size, stride, padding, bias, activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft- len(ft_h_list) + i](ft_l)
            ft_fusion = ft_l

            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft-i-1](ft_fusion - ft_h_list[i]) + ft_l_list[len(ft_h_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                   ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]

                for j in range(self.num_ft - i):
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)

                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i+1):
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class J_submodel(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=128, use_dropout=False, padding_type='reflect'):
        super(J_submodel, self).__init__()

        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225
        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(True))
        self.layer2 = resnext.layer2
        self.layer3 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(True))

        self.RIRs1 = ResidualInResiduals(ngf)
        self.RIRs2 = ResidualInResiduals(ngf * 2)
        self.RIRs3 = ResidualInResiduals(ngf * 4)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf *4  , ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2 *2 , ngf , kernel_size=3, stride=2, padding=1, output_padding=1),
                                nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf  * 2, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

        self.fusion1 = Encoder_MDCBlock1(ngf * 2, 2, mode = 'iter2')
        self.fusion2 = Encoder_MDCBlock1(ngf * 4, 3, mode='iter2')
        self.fusion3 = Encoder_MDCBlock1(ngf * 8, 4, mode='iter2')

        self.dehaze = nn.Sequential()
        for i in range(0, 18):
            self.dehaze.add_module('res%d' % i, ResidualBlock(ngf * 8))

    def forward(self, input):


        x = (input - self.mean) / self.std
        x_down1 = self.layer0(x)
        feature_mem = [x_down1]
        x_RIR_1 = self.RIRs1(x_down1)

        x_down2 = self.layer1(x_RIR_1)
        x_down2 = self.fusion1(x_down2,feature_mem)
        feature_mem.append(x_down2)
        x_RIR_2 = self.RIRs2(x_down2)

        x_down3 = self.layer2(x_RIR_2)
        x_down3 = self.fusion2(x_down3, feature_mem)
        feature_mem.append(x_down3)
        x_RIR_3 = self.RIRs3(x_down3)

        x_up1 = self.up1(x_RIR_3)
        x_up2 = self.up2(torch.cat([x_up1, x_RIR_2], 1))
        predict1 = self.up3(torch.cat([x_up2, x_RIR_1], 1))

        return predict1

class dehaze(nn.Module):
  def __init__(self):
    super(dehaze, self).__init__()

    self.course_J = J_submodel()

  def forward(self, x):

    predict1 = self.course_J(x)
    predict2 = predict1

    return predict1,predict2
