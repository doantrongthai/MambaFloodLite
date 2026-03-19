import torch
import torch.nn as nn
import torch.nn.functional as F
from models.propose_model.module.axial_dw import AxialDW


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c // 2 + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AxialDWEncoderBlock(nn.Module):
    """
    Branch 1: AxialDW -> PW Conv -> BN -> ReLU -> skip
              MaxPool -> x
    """
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.pw    = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn    = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x    = self.axial(x)
        skip = self.relu(self.bn(self.pw(x)))
        x    = self.pool(skip)
        return x, skip


class UNet_AxialDW(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)

        self.e1 = AxialDWEncoderBlock(16,  16,  kernel_size=3)
        self.e2 = AxialDWEncoderBlock(16,  32,  kernel_size=3)
        self.e3 = AxialDWEncoderBlock(32,  64,  kernel_size=3)
        self.e4 = AxialDWEncoderBlock(64, 128,  kernel_size=3)

        self.bottleneck = DoubleConv(128, 256)

        self.d4 = DecoderBlock(256, 128, 128)
        self.d3 = DecoderBlock(128,  64,  64)
        self.d2 = DecoderBlock( 64,  32,  32)
        self.d1 = DecoderBlock( 32,  16,  16)

        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)

        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        x = self.bottleneck(x)

        x = self.d4(x,  skip4)
        x = self.d3(x,  skip3)
        x = self.d2(x,  skip2)
        x = self.d1(x,  skip1)

        return self.out(x)


def build_model(num_classes=1):
    return UNet_AxialDW(in_channels=3, num_classes=num_classes)

