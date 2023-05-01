#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:18:53 2023

@author: jiu7
"""
import torch
import torch.nn as nn
import math
from thop import profile, clever_format
from datetime import datetime

def get_decoder_block(in_channels, out_channels, scale):
    assert scale in [1/8, 1/4, 1/2, 1, 2, 4, 8, 'fusion']
    
    if scale == 'fusion':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
            )
    
    elif scale < 1:
        return nn.Sequential(
            nn.MaxPool2d(int(1/scale), int(1/scale), ceil_mode=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
            )
    
    elif scale == 1:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
            )
    
    elif scale > 1:
        return nn.Sequential(
            nn.Upsample(scale_factor = scale, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
            )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            # depthwise separable convolution block
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
            # Bottleneck with expansion layer
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
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Decoder(nn.Module):
    """
    Decoder block: upsample and concatenate with features maps from the encoder part
    """
    def __init__(self, up_in_c, cat_in_c, out_c, expand_ratio = 0.15):
        super().__init__()
        self.ir1 = InvertedResidual(up_in_c + cat_in_c, out_c, stride=1,expand_ratio=expand_ratio)

    def forward(self, up_in, cat_in): # feats from upsampling, and feats to cat
        x = self.ir1(torch.cat([up_in, cat_in] , dim=1))
        return x
 
        
class M2UNet_self(nn.Module):
        def __init__(self, upsamplemode='bilinear', in_channels=3, n_classes=1):
            super().__init__()
            # Encoder
            # 3->32, half the size
            self.in_conv = nn.Sequential( 
                nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True))
            self.encoder1 = InvertedResidual(inp = 32, oup = 16, stride = 1, expand_ratio = 1)
            self.down1 = InvertedResidual(inp = 16, oup = 24, stride = 2, expand_ratio = 6)
            self.encoder2 = InvertedResidual(inp = 24, oup = 24, stride = 1, expand_ratio = 6)
            self.down2 = InvertedResidual(inp = 24, oup = 32, stride = 2, expand_ratio = 6)
            self.encoder3 = nn.Sequential(
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6)
                )
            self.down3 = InvertedResidual(inp = 32, oup = 64, stride = 2, expand_ratio = 6)
            self.encoder4 = nn.Sequential(
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6)
                )
            
            # Decoder
            self.upsampler = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False)
            self.decoder1 = Decoder(up_in_c = 96, cat_in_c = 32, out_c = 64, expand_ratio = 0.15) # half the channels
            self.decoder2 = Decoder(up_in_c = 64, cat_in_c = 24, out_c = 44, expand_ratio = 0.15) # half the channels
            self.decoder3 = Decoder(up_in_c = 44, cat_in_c = 16, out_c = 30, expand_ratio = 0.15) # half the channels
            self.out_conv = Decoder(up_in_c = 30, cat_in_c = 3, out_c = 1, expand_ratio = 0.15)
            
            # initilaize weights 
            self._initialize_weights()

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
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
            
        def forward(self, x):
            x1 = self.encoder1(self.in_conv(x)) #
            x2 = self.down1(x1)
            x3 = self.encoder2(x2) #
            x4 = self.down2(x3)
            x5 = self.encoder3(x4) #
            x6 = self.down3(x5)
            x7 = self.encoder4(x6) 
            
            x8 = self.upsampler(x7)
            x9 = self.decoder1(x8, x5)
            x10 = self.upsampler(x9)
            x11 = self.decoder2(x10, x3)
            x12 = self.upsampler(x11)
            x13 = self.decoder3(x12, x1)
            x14 = self.upsampler(x13)
            x15 = self.out_conv(x14, x)
            
            return x15

class M2UNet_3plus(nn.Module):
        def __init__(self, upsamplemode='bilinear', in_channels=3, n_classes=1):
            super().__init__()
            # Encoder
            # 3->32, half the size
            self.in_conv = nn.Sequential( 
                nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True))
            self.encoder1 = InvertedResidual(inp = 32, oup = 16, stride = 1, expand_ratio = 1)
            self.down1 = InvertedResidual(inp = 16, oup = 24, stride = 2, expand_ratio = 6)
            self.encoder2 = InvertedResidual(inp = 24, oup = 24, stride = 1, expand_ratio = 6)
            self.down2 = InvertedResidual(inp = 24, oup = 32, stride = 2, expand_ratio = 6)
            self.encoder3 = nn.Sequential(
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6)
                )
            self.down3 = InvertedResidual(inp = 32, oup = 64, stride = 2, expand_ratio = 6)
            self.encoder4 = nn.Sequential(
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6)
                )
            
            # Decoder
            self.upsampler = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False)
            filters = [16, 24, 32, 96]
            self.CatChannels = 8
            self.UpChannels = 32
            self.h1_PT_hd3 = get_decoder_block(filters[0], self.CatChannels, 1/4)
            self.h2_PT_hd3 = get_decoder_block(filters[1], self.CatChannels, 1/2)
            self.h3_Cat_hd3 = get_decoder_block(filters[2], self.CatChannels, 1)
            self.hd4_UT_hd3 = get_decoder_block(filters[3], self.CatChannels, 2)
            self.fusion3d_1 = get_decoder_block(self.UpChannels, self.UpChannels, 'fusion')
            '''stage 2d '''
            self.h1_PT_hd2 = get_decoder_block(filters[0], self.CatChannels, 1/2)
            self.h2_Cat_hd2 = get_decoder_block(filters[1], self.CatChannels, 1)
            self.hd3_UT_hd2 = get_decoder_block(self.UpChannels, self.CatChannels, 2)
            self.hd4_UT_hd2 = get_decoder_block(filters[3], self.CatChannels, 4)
            self.fusion2d_1 = get_decoder_block(self.UpChannels, self.UpChannels, 'fusion')
            '''stage 1d'''
            self.h1_Cat_hd1 = get_decoder_block(filters[0], self.CatChannels, 1)
            self.hd2_UT_hd1 = get_decoder_block(self.UpChannels, self.CatChannels, 2)
            self.hd3_UT_hd1 = get_decoder_block(self.UpChannels, self.CatChannels, 4)
            self.hd4_UT_hd1 = get_decoder_block(filters[3], self.CatChannels, 8)
            self.fusion1d_1 = get_decoder_block(self.UpChannels, self.UpChannels, 'fusion')   
            
            self.out_conv = Decoder(up_in_c = 32, cat_in_c = 3, out_c = 1, expand_ratio = 0.15)
            
            # initilaize weights 
            self._initialize_weights()

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
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
            
        def forward(self, x):
            x1 = self.encoder1(self.in_conv(x)) #
            x2 = self.down1(x1)
            x3 = self.encoder2(x2) #
            x4 = self.down2(x3)
            x5 = self.encoder3(x4) #
            x6 = self.down3(x5)
            x7 = self.encoder4(x6) 
            
            h1_PT_hd3 = self.h1_PT_hd3(x1)
            h2_PT_hd3 = self.h2_PT_hd3(x3)
            h3_Cat_hd3 = self.h3_Cat_hd3(x5)
            hd4_UT_hd3 = self.hd4_UT_hd3(x7)
            hd3 = self.fusion3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))
            
            h1_PT_hd2 = self.h1_PT_hd2(x1)
            h2_Cat_hd2 = self.h2_Cat_hd2(x3)
            hd3_UT_hd2 = self.hd3_UT_hd2(hd3)
            hd4_UT_hd2 = self.hd4_UT_hd2(x7)
            hd2 = self.fusion2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1))
            
            h1_Cat_hd1 = self.h1_Cat_hd1(x1)
            hd2_UT_hd1 = self.hd2_UT_hd1(hd2) 
            hd3_UT_hd1 = self.hd3_UT_hd1(hd3) 
            hd4_UT_hd1 = self.hd4_UT_hd1(x7) 
            hd1 = self.fusion1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1)) 
            
            x14 = self.upsampler(hd1)
            x15 = self.out_conv(x14, x)
            
            return x15


def get_dw_decoder_block(in_channels, out_channels, scale):
    assert scale in [1/8, 1/4, 1/2, 1, 2, 4, 8, 'fusion']
    
    if scale == 'fusion':
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
            )
    
    elif scale < 1:
        return nn.Sequential(
            nn.MaxPool2d(int(1/scale), int(1/scale), ceil_mode=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
            )
    
    elif scale == 1:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
            )
    
    elif scale > 1:
        return nn.Sequential(
            nn.Upsample(scale_factor = scale, mode='bilinear'),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
            )
        
class M2UNet_3plus_dw(nn.Module):
        def __init__(self, upsamplemode='bilinear', in_channels=3, n_classes=1):
            super().__init__()
            # Encoder
            # 3->32, half the size
            self.in_conv = nn.Sequential( 
                nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True))
            self.encoder1 = InvertedResidual(inp = 32, oup = 16, stride = 1, expand_ratio = 1)
            self.down1 = InvertedResidual(inp = 16, oup = 24, stride = 2, expand_ratio = 6)
            self.encoder2 = InvertedResidual(inp = 24, oup = 24, stride = 1, expand_ratio = 6)
            self.down2 = InvertedResidual(inp = 24, oup = 32, stride = 2, expand_ratio = 6)
            self.encoder3 = nn.Sequential(
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6)
                )
            self.down3 = InvertedResidual(inp = 32, oup = 64, stride = 2, expand_ratio = 6)
            self.encoder4 = nn.Sequential(
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6)
                )
            
            # Decoder
            self.upsampler = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False)
            filters = [16, 24, 32, 96]
            self.CatChannels = 8
            self.UpChannels = 32
            self.h1_PT_hd3 = get_dw_decoder_block(filters[0], self.CatChannels, 1/4)
            self.h2_PT_hd3 = get_dw_decoder_block(filters[1], self.CatChannels, 1/2)
            self.h3_Cat_hd3 = get_dw_decoder_block(filters[2], self.CatChannels, 1)
            self.hd4_UT_hd3 = get_dw_decoder_block(filters[3], self.CatChannels, 2)
            self.fusion3d_1 = get_dw_decoder_block(self.UpChannels, self.UpChannels, 'fusion')
            '''stage 2d '''
            self.h1_PT_hd2 = get_dw_decoder_block(filters[0], self.CatChannels, 1/2)
            self.h2_Cat_hd2 = get_dw_decoder_block(filters[1], self.CatChannels, 1)
            self.hd3_UT_hd2 = get_dw_decoder_block(self.UpChannels, self.CatChannels, 2)
            self.hd4_UT_hd2 = get_dw_decoder_block(filters[3], self.CatChannels, 4)
            self.fusion2d_1 = get_dw_decoder_block(self.UpChannels, self.UpChannels, 'fusion')
            '''stage 1d'''
            self.ori_PT_hd1 = get_dw_decoder_block(3, self.CatChannels, 1/2)
            self.h1_Cat_hd1 = get_dw_decoder_block(filters[0], self.CatChannels, 1)
            self.hd2_UT_hd1 = get_dw_decoder_block(self.UpChannels, self.CatChannels, 2)
            self.hd3_UT_hd1 = get_dw_decoder_block(self.UpChannels, self.CatChannels, 4)
            self.hd4_UT_hd1 = get_dw_decoder_block(filters[3], self.CatChannels, 8)
            self.fusion1d_1 = get_dw_decoder_block(5 * self.CatChannels, self.UpChannels, 'fusion')   
            
            self.out_conv = Decoder(up_in_c = 32, cat_in_c = 3, out_c = 1, expand_ratio = 0.15)
            
            # initilaize weights 
            self._initialize_weights()

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
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
            
        def forward(self, x):
            x1 = self.encoder1(self.in_conv(x)) #
            x2 = self.down1(x1)
            x3 = self.encoder2(x2) #
            x4 = self.down2(x3)
            x5 = self.encoder3(x4) #
            x6 = self.down3(x5)
            x7 = self.encoder4(x6) 
            
            h1_PT_hd3 = self.h1_PT_hd3(x1)
            h2_PT_hd3 = self.h2_PT_hd3(x3)
            h3_Cat_hd3 = self.h3_Cat_hd3(x5)
            hd4_UT_hd3 = self.hd4_UT_hd3(x7)
            hd3 = self.fusion3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))
            
            h1_PT_hd2 = self.h1_PT_hd2(x1)
            h2_Cat_hd2 = self.h2_Cat_hd2(x3)
            hd3_UT_hd2 = self.hd3_UT_hd2(hd3)
            hd4_UT_hd2 = self.hd4_UT_hd2(x7)
            hd2 = self.fusion2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1))
            
            ori_PT_hd1 = self.ori_PT_hd1(x)
            h1_Cat_hd1 = self.h1_Cat_hd1(x1)
            hd2_UT_hd1 = self.hd2_UT_hd1(hd2) 
            hd3_UT_hd1 = self.hd3_UT_hd1(hd3) 
            hd4_UT_hd1 = self.hd4_UT_hd1(x7) 
            hd1 = self.fusion1d_1(torch.cat((ori_PT_hd1, h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1)) 
            
            x14 = self.upsampler(hd1)
            x15 = self.out_conv(x14, x)
            
            return x15        

class M2UNet_3plus_dw_deepsup(nn.Module):
        def __init__(self, upsamplemode='bilinear', in_channels=3, n_classes=1):
            super().__init__()
            # Encoder
            # 3->32, half the size
            self.in_conv = nn.Sequential( 
                nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU6(inplace=True))
            self.encoder1 = InvertedResidual(inp = 32, oup = 16, stride = 1, expand_ratio = 1)
            self.down1 = InvertedResidual(inp = 16, oup = 24, stride = 2, expand_ratio = 6)
            self.encoder2 = InvertedResidual(inp = 24, oup = 24, stride = 1, expand_ratio = 6)
            self.down2 = InvertedResidual(inp = 24, oup = 32, stride = 2, expand_ratio = 6)
            self.encoder3 = nn.Sequential(
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 32, oup = 32, stride = 1, expand_ratio = 6)
                )
            self.down3 = InvertedResidual(inp = 32, oup = 64, stride = 2, expand_ratio = 6)
            self.encoder4 = nn.Sequential(
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 64, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 64, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6),
                InvertedResidual(inp = 96, oup = 96, stride = 1, expand_ratio = 6)
                )
            
            # Decoder
            self.upsampler = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False)
            filters = [16, 24, 32, 96]
            self.CatChannels = 8
            self.UpChannels = 32
            self.h1_PT_hd3 = get_dw_decoder_block(filters[0], self.CatChannels, 1/4)
            self.h2_PT_hd3 = get_dw_decoder_block(filters[1], self.CatChannels, 1/2)
            self.h3_Cat_hd3 = get_dw_decoder_block(filters[2], self.CatChannels, 1)
            self.hd4_UT_hd3 = get_dw_decoder_block(filters[3], self.CatChannels, 2)
            self.fusion3d_1 = get_dw_decoder_block(self.UpChannels, self.UpChannels, 'fusion')
            '''stage 2d '''
            self.h1_PT_hd2 = get_dw_decoder_block(filters[0], self.CatChannels, 1/2)
            self.h2_Cat_hd2 = get_dw_decoder_block(filters[1], self.CatChannels, 1)
            self.hd3_UT_hd2 = get_dw_decoder_block(self.UpChannels, self.CatChannels, 2)
            self.hd4_UT_hd2 = get_dw_decoder_block(filters[3], self.CatChannels, 4)
            self.fusion2d_1 = get_dw_decoder_block(self.UpChannels, self.UpChannels, 'fusion')
            '''stage 1d'''
            self.ori_PT_hd1 = get_dw_decoder_block(3, self.CatChannels, 1/2)
            self.h1_Cat_hd1 = get_dw_decoder_block(filters[0], self.CatChannels, 1)
            self.hd2_UT_hd1 = get_dw_decoder_block(self.UpChannels, self.CatChannels, 2)
            self.hd3_UT_hd1 = get_dw_decoder_block(self.UpChannels, self.CatChannels, 4)
            self.hd4_UT_hd1 = get_dw_decoder_block(filters[3], self.CatChannels, 8)
            self.fusion1d_1 = get_dw_decoder_block(4 * self.CatChannels, self.UpChannels, 'fusion')   
            
            self.out_conv = Decoder(up_in_c = 32, cat_in_c = 3, out_c = 1, expand_ratio = 0.15)
            
            # DeepSup
            self.deepsup5 = nn.Conv2d(35, n_classes, 3, padding = 1)
            self.deepsup4 = nn.Sequential(
                nn.Conv2d(self.UpChannels, n_classes, 3, padding = 1),
                nn.Upsample(scale_factor=2, mode='bilinear')
                )
            self.deepsup3 = nn.Sequential(
                nn.Conv2d(self.UpChannels, n_classes, 3, padding = 1),
                nn.Upsample(scale_factor=4, mode='bilinear')
                )
            self.deepsup2 = nn.Sequential(
                nn.Conv2d(self.UpChannels, n_classes, 3, padding = 1),
                nn.Upsample(scale_factor=8, mode='bilinear')
                )
            self.deepsup1 = nn.Sequential(
                nn.Conv2d(filters[3], n_classes, 3, padding = 1),
                nn.Upsample(scale_factor=16, mode='bilinear')
                )
                                          
            # initilaize weights 
            self._initialize_weights()

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
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
            
        def forward(self, x, train = True):
            x1 = self.encoder1(self.in_conv(x)) #
            x2 = self.down1(x1)
            x3 = self.encoder2(x2) #
            x4 = self.down2(x3)
            x5 = self.encoder3(x4) #
            x6 = self.down3(x5)
            x7 = self.encoder4(x6)
            
            h1_PT_hd3 = self.h1_PT_hd3(x1)
            h2_PT_hd3 = self.h2_PT_hd3(x3)
            h3_Cat_hd3 = self.h3_Cat_hd3(x5)
            hd4_UT_hd3 = self.hd4_UT_hd3(x7)
            hd3 = self.fusion3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))
            
            h1_PT_hd2 = self.h1_PT_hd2(x1)
            h2_Cat_hd2 = self.h2_Cat_hd2(x3)
            hd3_UT_hd2 = self.hd3_UT_hd2(hd3)
            hd4_UT_hd2 = self.hd4_UT_hd2(x7)
            hd2 = self.fusion2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1))
            
            ori_PT_hd1 = self.ori_PT_hd1(x)
            h1_Cat_hd1 = self.h1_Cat_hd1(x1)
            hd2_UT_hd1 = self.hd2_UT_hd1(hd2) 
            hd3_UT_hd1 = self.hd3_UT_hd1(hd3) 
            #hd4_UT_hd1 = self.hd4_UT_hd1(x7) 
            hd1 = self.fusion1d_1(torch.cat((ori_PT_hd1, h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1), 1)) 
            
            x14 = self.upsampler(hd1)
            xout = self.out_conv(x14, x)
            
            if train:
                deepsup1 = self.deepsup1(x7)
                deepsup2 = self.deepsup2(hd3)
                deepsup3 = self.deepsup3(hd2)
                deepsup4 = self.deepsup4(hd1)
                deepsup5 = self.deepsup5(torch.cat((x14, x), 1))
                deepsup1, deepsup2, deepsup3, deepsup4, deepsup5 = \
                torch.sigmoid(deepsup1), torch.sigmoid(deepsup2), torch.sigmoid(deepsup3), \
                    torch.sigmoid(deepsup4), torch.sigmoid(deepsup5)
                return xout, deepsup1, deepsup2, deepsup3, deepsup4, deepsup5        
            else:
                return xout
       