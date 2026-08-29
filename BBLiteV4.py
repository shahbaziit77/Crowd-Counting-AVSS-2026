import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

def conv_bnr_oup(inp, oup, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=1, padding=padding, groups=oup, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )
    
def conv_bnr_inp(inp, oup, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=1, padding=padding, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

def conv_bnr_oup1X1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

def conv_bnrDS_oup(inp, oup, kernel_size, padding):
    return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=2, padding=padding, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

def conv_bnrDS_inp(inp, oup, kernel_size, padding):
    return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=0.1,inplace=True)
    )

        
class inception(nn.Module):
    def __init__(self, in_channel=32):
        super(inception,self).__init__()
        self.conv = conv_bnr_oup(in_channel, in_channel, 3, 1)
        self.conv1 = conv_bnr_oup1X1(in_channel, in_channel//2)
        self.conv2 = conv_bnr_oup(in_channel, in_channel//2, 3, 1)
        self.conv3 = conv_bnr_oup(in_channel, in_channel, 3, 1)
        #self.shuffle = Shuffling(in_channel)
        
    def forward(self, input):
        x = self.conv(input)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv3(out)
        #out = self.shuffle(out)
        out = out + input
        
        return out
        
class inceptionDS(nn.Module):
    def __init__(self, in_channel=32):
        super(inceptionDS,self).__init__()
        self.conv = conv_bnrDS_oup(in_channel, in_channel, 3, 1)
        self.conv_down = conv_bnrDS_inp(in_channel, 2*in_channel, 3, 1)
        
        self.conv1 = conv_bnr_oup1X1(in_channel, in_channel//2)
        self.conv2 = conv_bnr_oup(in_channel, in_channel//2, 3, 1)
        self.conv3 = conv_bnr_inp(in_channel, 2*in_channel, 3, 1)
        
    def forward(self, input):
        down = self.conv_down(input)
        x = self.conv(input)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv3(out)
        out = out + down
        
        return out
        

class BBLiteV4(nn.Module):
    def __init__(self):
        super(BBLiteV4, self).__init__()  
        self.conv1 = conv_bn(3, 8, 2, leaky = 0.1)
        self.conv2 = conv_dw(8, 16, 1)
        self.conv3 = conv_dw(16, 32, 2)
        
        self.stage1_1 = inceptionDS(32)
        self.stage1_2 = inception(64)
        self.stage1_3 = inception(64)
        self.stage1 = inception(64)   

        self.stage2_1 = inceptionDS(64)
        self.stage2_2 = inception(128)
        self.stage2_3 = inception(128)
        self.stage2 = inception(128)

        self.stage3_1 = inceptionDS(128)
        self.stage3_2 = inception(256)
        self.stage3_3 = inception(256)
        self.stage3 = inception(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        #x = self.conv_GNN(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage1_3(x)
        x = self.stage1(x)
        
        x = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage2_3(x)
        x = self.stage2(x)
        
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.stage3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
