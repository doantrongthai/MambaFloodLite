import torch
import torch.nn as nn
from ..module.axial_dw import AxialDW
from ..module.dual_vss_block import DualVSSBlock


class TripleBranchEncoderBlock(nn.Module):

    def __init__(self, in_c, out_c, kernel_size=7):
        super().__init__()
        
        self.branch1_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.branch1_pw = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.branch1_bn = nn.BatchNorm2d(in_c)
        self.branch1_relu = nn.ReLU(inplace=True)
        
        self.branch2_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.branch2_pw = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.branch2_bn = nn.BatchNorm2d(in_c)
        self.branch2_relu = nn.ReLU(inplace=True)
        self.branch2_vss = DualVSSBlock(hidden_dim=in_c, d_state=8)
        
        self.branch3_avgpool = nn.AdaptiveAvgPool2d(1)
        self.branch3_dw = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.branch3_sigmoid = nn.Sigmoid()
        
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_c * 3, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        b1 = self.branch1_axial(x)
        b1 = self.branch1_pw(b1)
        b1 = self.branch1_bn(b1)
        b1 = self.branch1_relu(b1)
        
        b2 = self.branch2_axial(x)
        b2 = self.branch2_pw(b2)
        b2 = self.branch2_bn(b2)
        b2 = self.branch2_relu(b2)
        B, C, H, W = b2.shape
        b2 = b2.permute(0, 2, 3, 1)
        b2 = self.branch2_vss(b2)
        b2 = b2.permute(0, 3, 1, 2)
        
        b3_avg = self.branch3_avgpool(x)
        b3_dw = self.branch3_dw(b3_avg)
        b3_attn = self.branch3_sigmoid(b3_dw)
        b3 = x * b3_attn
        
        concat = torch.cat([b1, b2, b3], dim=1)
        
        skip = self.channel_adjust(concat)
        
        x = self.pool(skip)
        
        return x, skip