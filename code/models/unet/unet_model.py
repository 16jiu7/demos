""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        # this in_channels is in_channels of main branch plus concat branch
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): # x2 is the features from encoder, to be concated 
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2] #H
        diffX = x2.size()[3] - x1.size()[3] #W

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # enter list of 4 to pad the last 2 dimensions of x1
        
        # this just means padding x1 so that it has the same size as x2, and make sure we get int in size
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, channels = [64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, channels[0]))
        self.down1 = (Down(channels[0], channels[1]))
        self.down2 = (Down(channels[1], channels[2]))
        self.down3 = (Down(channels[2], channels[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(channels[3], channels[4] // factor)) 
        # if bilinear than we just don't get to 1024 channels in the end
        self.up1 = (Up(channels[4], channels[3] // factor, bilinear))
        self.up2 = (Up(channels[3], channels[2] // factor, bilinear))
        self.up3 = (Up(channels[2], channels[1] // factor, bilinear))
        self.up4 = (Up(channels[1], channels[0], bilinear))
        self.outc = (OutConv(channels[0], n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def get_concat_feats(self, x):
        # get concated multiscale feats, without CNN pred result
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        resizer1 = nn.Upsample(size = x1.shape[2:], mode='bilinear', align_corners=True)
        resizer2 = nn.Upsample(size = x1.shape[2:], mode='bilinear', align_corners=True)
        resizer3 = nn.Upsample(size = x1.shape[2:], mode='bilinear', align_corners=True)
        resizer4 = nn.Upsample(size = x1.shape[2:], mode='bilinear', align_corners=True)
        feats = torch.cat([x1, resizer1(x2), resizer2(x3), resizer3(x4), resizer4(x5)], dim = 1)
        return feats # shape = batch_size * sum(channels) * H * W
    
    def run_encoder(self, x):
        # only get downscaled feats, without CNN pred result
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5 # shape = batch_size * max(channels) * H/16 * W/16

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
        

    
