# Importing the libraries  
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.models._utils as _utils  

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),   
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):                                    
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky, inplace=True),

        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),   
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky, inplace=True),  
    )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Context Module (CM): Single Shot Headless (SSH) Detector 
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):   
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)   

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Feature Pyramid Network (FPN)
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):   
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)   

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # names = list(input.keys())
        #input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2]) 
        
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        # out = output1
        return out  
        
#============================================================================================================        
        
# Path Aggregation Network (PANet)
class PANet(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PANet, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
            
        # Top-down Pathway (FPN style)
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)   
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)
        
        # Bottom-up Pathway (PANet style)
        self.down1 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)
        self.down2 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)

    def forward(self, input):
        # Convert input dictionary to list
        # names = list(input.keys())
        #input = list(input.values())
        
        # Top-down Pathway
        output1 = self.output1(input[0])
        output2_1 = self.output2(input[1])
        output3 = self.output3(input[2]) 
        
        up3 = F.interpolate(output3, size=[output2_1.size(2), output2_1.size(3)], mode="nearest")
        output2 = output2_1 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        # Bottom-up Pathway
        down1 = F.interpolate(output1, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + down1 + output2_1
        output2 = self.down1(output2)
        
        down2 = F.interpolate(output2, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + down2 
        output3 = self.down2(output3)
        
        # Outputs
        out = [output1, output2, output3]
        return out
        
#==========================================================================================================        

# PANet      
class PANet2(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PANet2, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
            
        # Top-down Pathway (FPN style)
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)   
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)
        
        # Bottom-up Pathway (PANet style)
        self.down1 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)
        self.down2 = conv_bn(out_channels, out_channels, stride=1, leaky=leaky)

    def forward(self, input):
        # Convert input dictionary to list
        # names = list(input.keys())
        #input = list(input.values())
        
        # Top-down Pathway
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2]) 
        
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        # Bottom-up Pathway
        down1 = F.interpolate(output1, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + down1 
        output2 = self.down1(output2)
        
        down2 = F.interpolate(output2, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + down2 
        output3 = self.down2(output3)
        
        # Outputs
        out = [output1, output2, output3]
        return out  

#===========================================================================================================

# Bi-directional Feature Pyramid Network (BiFPN)
class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(BiFPN, self).__init__()
        self.panet1 = PANet(in_channels_list, out_channels)
        self.panet2 = PANet([out_channels, out_channels, out_channels], out_channels)
        
    def forward(self, input):
        x = self.panet1(input)
        x = self.panet2(x)   
        return x
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------
# MobileNetV1
#-------------------------------------------------------------------------------------

# MobileNetV1_0.25
class MobileNetV1_0_25(nn.Module):
    def __init__(self):
        super(MobileNetV1_0_25, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),    # 3  
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)  
        x = self.fc(x)
        return x

#===================================================================

# MobileNetV1_0.5
class MobileNetV1_0_5(nn.Module):
    def __init__(self):
        super(MobileNetV1_0_5, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 16, 2, leaky=0.1),    # 3  
            conv_dw(16, 32, 1),   # 7
            conv_dw(32, 64, 2),  # 11
            conv_dw(64, 64, 1),  # 19
            conv_dw(64, 128, 2),  # 27
            conv_dw(128, 128, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(128, 256, 2),  # 43 + 16 = 59
            conv_dw(256, 256, 1), # 59 + 32 = 91
            conv_dw(256, 256, 1), # 91 + 32 = 123
            conv_dw(256, 256, 1), # 123 + 32 = 155
            conv_dw(256, 256, 1), # 155 + 32 = 187
            conv_dw(256, 256, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(256, 512, 2), # 219 +3 2 = 241
            conv_dw(512, 512, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 512)  
        x = self.fc(x)
        return x  

#====================================================================

# MobileNetV1_1.0
class MobileNetV1_1_0(nn.Module):
    def __init__(self):
        super(MobileNetV1_1_0, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 2, leaky=0.1),    # 3  
            conv_dw(32, 64, 1),   # 7
            conv_dw(64, 128, 2),  # 11
            conv_dw(128, 128, 1),  # 19
            conv_dw(128, 256, 2),  # 27
            conv_dw(256, 256, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),  # 43 + 16 = 59
            conv_dw(512, 512, 1), # 59 + 32 = 91
            conv_dw(512, 512, 1), # 91 + 32 = 123
            conv_dw(512, 512, 1), # 123 + 32 = 155
            conv_dw(512, 512, 1), # 155 + 32 = 187
            conv_dw(512, 512, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2), # 219 +3 2 = 241
            conv_dw(1024, 1024, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)  
        x = self.fc(x)
        return x  

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define a simple fully connected (FC) block (FC -> Batch Norm -> Dropout -> ReLU)
class SimpleFC(nn.Module):
    def __init__(self, in_features, out_features, dropout):
    #def __init__(self, in_features, out_features, dropout, leaky=0.1):
        super(SimpleFC, self).__init__()
        # Flatten the image (H x W x 3) into a vector
        #self.flatten = nn.Flatten()
        
        # Fully connected (FC) layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),    # FC
            nn.BatchNorm1d(out_features),                                     # Batch Normalization for FC
            nn.Dropout(p=dropout),                                            # Dropout with p=dropout
            nn.ReLU(inplace=True)                                             # Apply ReLU activation
            #nn.LeakyReLU(negative_slope=leaky, inplace=True)                 # Apply Leaky ReLU activation                                             
        )
        
        # Reshape the output to the original image resolution (H x W x 3)
        #self.unflatten = nn.Unflatten(1, (3, H, W))
        
    def forward(self, x):
        # Flatten the input: (batch_size, 3, H, W) -> (batch_size, 3 * H * W)
        #x = x.view(x.size(0), -1)   # x.size(0) is the batch size
        #x = self.flatten(x)         # Flatten the input
        #out = self.fc(x)             # Pass through fc
        x = self.fc(x)               # Pass through fc
        
        # Unflatten the output back to (batch_size, 3, H, W)
        #x = out.view(out.size(0), x.size(1), x.size(2), x.size(3))  # Reshape to (batch_size, 3, H, W)
        #x = self.unflatten(out)               # Reshape to (3, H, W)
        
        return x

#============================================================================================================

# Define a simple fully connected (FC) block without both the batch normalization and dropout (FC -> ReLU)
class SimpleFCWithoutBatchNormDropout(nn.Module):
    def __init__(self, in_features, out_features):
    #def __init__(self, in_features, out_features, leaky=0.1):
    #def __init__(self, in_features, out_features, dropout):
        super(SimpleFCWithoutBatchNormDropout, self).__init__()
        # Flatten the image (H x W x 3) into a vector
        #self.flatten = nn.Flatten()
        
        # Fully connected (FC) layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=True),    # FC  
            #nn.BatchNorm1d(out_features),                                    # Batch Normalization for FC
            #nn.Dropout(p=dropout),                                           # Dropout with p=dropout
            nn.ReLU(inplace=True)                                             # Apply ReLU activation
            #nn.LeakyReLU(negative_slope=leaky, inplace=True)                 # Apply Leaky ReLU activation                                             
        )   
        
        # Reshape the output to the original image resolution (H x W x 3)
        #self.unflatten = nn.Unflatten(1, (3, H, W))
        
    def forward(self, x):
        # Flatten the input: (batch_size, 3, H, W) -> (batch_size, 3 * H * W)
        #x = x.view(x.size(0), -1)   # x.size(0) is the batch size
        #x = self.flatten(x)         # Flatten the input
        #out = self.fc(x)             # Pass through fc
        x = self.fc(x)               # Pass through fc
        
        # Unflatten the output back to (batch_size, 3, H, W)
        #x = out.view(out.size(0), x.size(1), x.size(2), x.size(3))  # Reshape to (batch_size, 3, H, W)
        #x = self.unflatten(out)               # Reshape to (3, H, W)
        
        return x   
    
#=============================================================================================================

# Define a simple fully connected (FC) block without ReLU (FC)
class SimpleFCWithoutReLU(nn.Module):
    def __init__(self, in_features, out_features):
    #def __init__(self, in_features, out_features, leaky=0.1):
    #def __init__(self, in_features, out_features, dropout):
        super(SimpleFCWithoutReLU, self).__init__()
        # Flatten the image (H x W x 3) into a vector
        #self.flatten = nn.Flatten()
        
        # Fully connected (FC) layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=True),    # FC
            #nn.BatchNorm1d(out_features),                                    # Batch Normalization for FC
            #nn.Dropout(p=dropout),                                           # Dropout with p=dropout
            #nn.ReLU(inplace=True)                                            # Apply ReLU activation
            #nn.LeakyReLU(negative_slope=leaky, inplace=True)                 # Apply Leaky ReLU activation                                             
        )
        
        # Reshape the output to the original image resolution (H x W x 3)
        #self.unflatten = nn.Unflatten(1, (3, H, W))
        
    def forward(self, x):
        # Flatten the input: (batch_size, 3, H, W) -> (batch_size, 3 * H * W)
        #x = x.view(x.size(0), -1)   # x.size(0) is the batch size
        #x = self.flatten(x)         # Flatten the input
        #out = self.fc(x)             # Pass through fc
        x = self.fc(x)               # Pass through fc
        
        # Unflatten the output back to (batch_size, 3, H, W)
        #x = out.view(out.size(0), x.size(1), x.size(2), x.size(3))  # Reshape to (batch_size, 3, H, W)
        #x = self.unflatten(out)               # Reshape to (3, H, W)
        
        return x   
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define a basic convolution block (Conv -> Batch Norm -> ReLU)
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    #def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leaky=0.1):
        super(BasicConv, self).__init__()
        # Convolutional (Conv) layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),     # Conv
            #nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),               # Conv
            nn.BatchNorm2d(out_channels),                                                                                                # Batch Normalization for Conv
            nn.ReLU(inplace=True)                                                                                                        # Apply ReLU activation
            #nn.LeakyReLU(negative_slope=leaky, inplace=True)                                                                            # Apply Leaky ReLU activation
        )

    def forward(self, x):
        x = self.conv(x)
        return x  
        
#=================================================================================================================        
        
# Define a basic convolution block without a batch normalization (Conv -> ReLU)
class BasicConvWithoutBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    #def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leaky=0.1):
        super(BasicConvWithoutBatchNorm, self).__init__()
        # Convolutional (Conv) layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),      # Conv
            #nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),                # Conv
            #nn.BatchNorm2d(out_channels),                                                                                               # Batch Normalization for Conv
            nn.ReLU(inplace=True)                                                                                                        # Apply ReLU activation
            #nn.LeakyReLU(negative_slope=leaky, inplace=True)                                                                            # Apply Leaky ReLU activation
        )

    def forward(self, x):
        x = self.conv(x)
        return x          
        
#==================================================================================================================
        
# Define a basic convolution block without ReLU (Conv)
class BasicConvWithoutReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConvWithoutReLU, self).__init__()
        # Convolutional (Conv) layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),     # Conv
            #nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),               # Conv
            #nn.BatchNorm2d(out_channels),                                                                                              # Batch Normalization for Conv
            #nn.ReLU(inplace=True)                                                                                                      # Apply ReLU activation
        )

    def forward(self, x):
        x = self.conv(x)
        return x   
        
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
# Channel Attention Module (CAM)  
class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):  
        super(CAM, self).__init__()  
        # Max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)
        
        # Global Average Pooling (GAP)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
         
        # Fully connected (FC) layer
        self.fc = nn.Sequential(
            SimpleFCWithoutBatchNormDropout(in_features=in_channels, out_features=in_channels // reduction_ratio),
            SimpleFCWithoutReLU(in_features=in_channels // reduction_ratio, out_features=in_channels)
            )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Max pooling layer
        #max = self.max_pool(x)
        max = F.adaptive_max_pool2d(x, output_size=1)
        
        # Global Average Pooling (GAP)
        #avg = self.avg_pool(x)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        
        # Global Average Pooling (GAP): Take mean across height and width dimensions
        #avg = torch.mean(x, dim=(2, 3), keepdim=True)
        
        b, c, _, _ = x.size()
        
        fc_max = self.fc(max.view(b, c)).view(b, c, 1, 1)
        fc_avg = self.fc(avg.view(b, c)).view(b, c, 1, 1)
        channel_attention = fc_max + fc_avg
        #channel_attention = self.sigmoid(channel_attention)
        channel_attention = F.sigmoid(channel_attention)
        # channel_attention = self.activation(channel_attention)
        
        # Apply the channel attention: Element-wise multiplication (i.e., Hadamard product)   
        x = x * channel_attention
        
        return x
        

# Spatial Attention Module (SAM)
class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        # Conv layer
        self.conv = BasicConvWithoutReLU(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)  
    
    def forward(self, x):
        max = torch.max(x, dim=1)[0].unsqueeze(dim=1)     # Take max across channel dimension and increment 1 channel dimension
        avg = torch.mean(x, dim=1).unsqueeze(dim=1)       # Take mean across channel dimension and increment 1 channel dimension  
        # Channel concatenate
        concat = torch.cat((max, avg), dim=1)   
        
        # Conv layer
        spatial_attention = self.conv(concat)
        
        # Apply the spatial attention: Element-wise multiplication (i.e., Hadamard product)
        x = x * spatial_attention
        
        return x       


# Convolutional Block Attention Module (CBAM)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CBAM, self).__init__()
        self.cam = CAM(in_channels=in_channels, reduction_ratio=reduction_ratio)
        self.sam = SAM()
        
    def forward(self, x):
        # Apply the channel attention
        output = self.cam(x)
        # Apply the spatial attention
        output = self.sam(output)
        # Apply the attention
        x = x + output  
        
        return x    

