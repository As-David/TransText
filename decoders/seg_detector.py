from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import math
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import imageio
BatchNorm2d = nn.BatchNorm2d


class ASPP(nn.Module):
    def __init__(self, in_channel=64, depth=32):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1)) #(1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




# Gated Feed-Forward Network (GFFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        self.hidden_features = int(dim*ffn_expansion_factor)

        self.project_in1 = nn.Conv2d(dim, self.hidden_features, kernel_size=1, bias=bias)
        self.project_in2 = nn.Conv2d(dim, self.hidden_features, kernel_size=1, bias=bias)
        
        self.dwconv1 = nn.Conv2d(self.hidden_features*2, self.hidden_features*2, kernel_size=3, stride=1, padding=1, groups=self.hidden_features*2, bias=bias)
        self.dwconv2 = nn.Conv2d(self.hidden_features*2, self.hidden_features, kernel_size=3, stride=1, padding=1, groups=self.hidden_features*2, bias=bias)
        self.dwconv1_2 = nn.Conv2d(self.hidden_features*2, self.hidden_features, kernel_size=3, stride=1, padding=1, groups=self.hidden_features*2, bias=bias)
        
        self.project_out1 = nn.Conv2d(self.hidden_features, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(self.hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, ref):
        x1 = self.project_in1(x)
        x2 = self.project_in2(x)
        
        x1 = self.dwconv1(x1)
        x1 = self.dwconv1_2(x1)
        x1 = F.gelu(x1)
        x1 = self.project_out1(x1)
        
        x2 = self.dwconv2(x2)
        x2 = ref.expand(-1, self.hidden_features, -1, -1).mul(x2)
        x2 = self.project_out2(x2)
        
        out = x1 + x2
        return out

# GPM
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input):
        x,ref=input
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x),ref)
        return (x,ref)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k=3,pad=1,stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=k, padding=pad,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ICM(nn.Module):
    def __init__(self, channels=32, r=4):
        super(ICM, self).__init__()
        out_channels = int(channels // r)
        self.dwconv1 = SeparableConv2d(channels,out_channels)
        
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, groups=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=13, padding=6, stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        return xlg

class FRM(nn.Module):
    def __init__(self, channels=32):
        super(FRM, self).__init__()
        self.aam1 = AAM(in_chan=64, out_chan=32 )
        self.aam2 = AAM(in_chan=64, out_chan=32 )
        self.aam3 = AAM(in_chan=64, out_chan=32 )
        self.icm = ICM(channels=32)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv1 = SeparableConv2d(channels, channels, k=3,pad=1,stride = 1)
        self.dwconv2 = SeparableConv2d(channels, channels, k=3,pad=1,stride= 1)
        self.dwconv3 = SeparableConv2d(channels, channels, k=3,pad=1,stride=1)
        self.dwconv4 = SeparableConv2d(channels, channels, k=3,pad=1,stride=1)
        self.dwconv5 = SeparableConv2d(channels, channels, k=3,pad=1,stride=1)
        self.dwconv6 = SeparableConv2d(channels, channels, k=3,pad=1,stride=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        
        self.b1 = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0)
        self.b2 = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0)
        self.b3 = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x1,x2,x3 ):
        x1_down = self.downsample(x1)
        x3_up = self.upsample(x3)
        branch1 = self.dwconv1(x1_down)
        branch2 = self.dwconv2(x2)
        branch3 = self.dwconv3(x3_up)
        fusion = branch1 + branch2 + branch3
        fusion = self.icm(fusion)
        branch1 = branch1 * self.sig(self.conv1(fusion))
        branch2 = branch2 * self.sig(self.conv2(fusion))
        branch3 = branch3 * self.sig(self.conv3(fusion))
        fusion = branch1 + branch2 + branch3
        
        branch1 = torch.cat([x1,self.upsample(self.dwconv4(fusion))],dim=1)
        branch1 = self.b1(branch1)
        branch2 = torch.cat([x2,self.dwconv5(fusion)],dim=1)
        branch2 = self.b2(branch2)
        branch3 = torch.cat([x3,self.downsample(self.dwconv6(fusion))],dim=1)
        branch3 = self.b3(branch3)
        return branch1,branch2,branch3

class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[256,512,1024,2048],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=True,
                 num_blocks = [4,6,6,8],
                 heads = [1,2,4,8],
                 ffn_expansion_factor = 2.66,
                 dim = 32,
                 LayerNorm_type = 'WithBias', 
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not. 
        smooth: If true, use bilinear instead of deconv. 
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=False)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=False)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=False)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=False)
       
        self.agg_conv = nn.Sequential(
            nn.Conv2d(96, 32, 1, padding=0, bias=False),
            BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1, bias=False))
        
        self.frm = FRM(channels = 32)
        self.ASPP1 =ASPP(in_channel=64, depth=64)
        self.ASPP2 =ASPP(in_channel=64, depth=64)
        
        self.ra5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1, bias=False)
        )
        self.ra4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1, bias=False)
        )
        self.ra3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1, bias=False)
        )
        
        self.agg_256chan = nn.Sequential(
            nn.Conv2d(32, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=.95),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(256, 256, 1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.out5 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(inner_channels//4, inner_channels //
                      4, 3, padding=1, bias=bias))
        self.out4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(inner_channels//4, inner_channels //
                      4, 3, padding=1, bias=bias))
        self.out3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(inner_channels//4, inner_channels //
                      4, 3, padding=1, bias=bias))
        self.out2 = nn.Conv2d(
            inner_channels//4, inner_channels//4, 3, padding=1, bias=bias)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            
        self.res2 = ConvBNReLU(in_channels[0],64)
        self.res3 = ConvBNReLU(in_channels[1],64)
        self.res4 = ConvBNReLU(in_channels[2],64)
        self.res5 = ConvBNReLU(in_channels[3],64)
        self.create_fusion = nn.Conv2d(96, 32, kernel_size=1, padding=0, bias=False)
        
        
        
        
        
        self.p4_cat = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.p3_cat = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.decoder_p5 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.decoder_p4 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_p3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_p2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.sigmoid = nn.Sigmoid()
        self.produce_binary = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, 2),
            BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, 2),
            BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 2, 2),
            nn.Sigmoid()
            )
        self.create_fusion_thresh = nn.Sequential(
            nn.Conv2d(96, 32, 1, padding=0, bias=False),
            BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 2, 2),
            BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            BatchNorm2d(32),
            nn.ReLU(inplace=True))
         
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=False),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)
        
        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    
    
    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
       
        in2 = self.res2(c2)
        in3 = self.res3(c3)
        in4 = self.res4(c4)
        in5 = self.res5(c5)
        
        in3,in4,in5 = self.frm(in3,in4,in5)
        
        p5 = self.out5(in5)  
        p4 = self.out4(in4)  
        p3 = self.out3(in3)  
        
        fusion = torch.cat([p3,p4,p5],dim=1)
        
        agg = self.agg_conv(fusion) 
        x_ref = F.interpolate(agg, scale_factor=8, mode='bilinear')
        x_ref_loss = self.sigmoid(x_ref)
        
        
        p5_ref = F.interpolate(agg, scale_factor=0.25, mode='bilinear')
        p5_ra = torch.sigmoid(p5_ref)
        p5_trans,p5_ra = self.decoder_p5((in5,p5_ra))
        p5_out_1chan = self.ra5(p5_trans)
        x_5 = p5_out_1chan + p5_ref     
        x_5_loss = F.interpolate(x_5, scale_factor=32, mode='bilinear')
        x_5_loss = self.sigmoid(x_5_loss)
       
        
        p5_trans_up = self.upsample(p5_trans)
        p4_cfp = self.p4_cat(torch.cat([p5_trans_up,in4],dim=1))
        p4_ref = F.interpolate(x_5, scale_factor=2, mode='bilinear')
        p4_ra = torch.sigmoid(p4_ref)
        p4_trans,p4_ra = self.decoder_p4((p4_cfp,p4_ra))
        p4_out_1chan = self.ra4(p4_trans)
        x_4 = p4_out_1chan + p4_ref    
        x_4_loss = F.interpolate(x_4, scale_factor=16, mode='bilinear')
        x_4_loss = self.sigmoid(x_4_loss)
        
        
        p4_trans_up = self.upsample(p4_trans)
        p3_cfp = self.p3_cat(torch.cat([p4_trans_up,in3],dim=1))
        p3_ref = F.interpolate(x_4, scale_factor=2, mode='bilinear')
        p3_ra = torch.sigmoid(p3_ref)
        p3_trans,p3_ra = self.decoder_p3((p3_cfp,p3_ra))
        
        
        binary = self.produce_binary(p3_trans) 
        fusion_thresh = self.create_fusion_thresh(fusion) 
        
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training : 
            if self.serial:
                fuse = torch.cat(
                        (fusion_thresh, nn.functional.interpolate(
                            binary, fusion_thresh.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary, x_ref_loss = x_ref_loss, x_5_loss=x_5_loss,x_4_loss=x_4_loss) 
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AAM(nn.Module):
    def __init__(self, in_chan, out_chan, reduction=32):
        super(AAM, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mi_chan = max(8, in_chan // reduction)

        self.conv1 = nn.Conv2d(in_chan//2, mi_chan, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mi_chan)
        self.act = h_swish()
        
        self.create_x1 = nn.Conv2d(in_chan, in_chan//2, kernel_size=1, stride=1, padding=0)
        self.create_x2 = nn.Conv2d(in_chan, in_chan//2, kernel_size=1, stride=1, padding=0)
        self.init = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.ReLU() )
        self.conv_h = nn.Conv2d(mi_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mi_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.final = nn.Sequential(
                nn.Conv2d(2*out_chan, out_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_chan),
                nn.ReLU() )
        
        
        self.channelatt = ChannelAttention(in_chan = in_chan, out_chan = out_chan)

    def forward(self, x,y):
        n,c,h,w = x.size()
        fusion = torch.cat([x,y],dim=1)
        fusion = self.init(fusion)
        x1 = self.create_x1(fusion)
        x2 = self.create_x2(fusion)
        identity = x1
        
        x_h = self.pool_h(x1)
        x_w = self.pool_w(x1).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out1 = identity * a_w * a_h
        out2 = self.channelatt(x2)
        out = torch.cat([out1,out2],dim=1)
        out = self.final(out)
        return out
