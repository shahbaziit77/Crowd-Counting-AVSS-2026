# Importing the libraries  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models         
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils  

from collections import OrderedDict

from model.net import MobileNetV1_0_25
from model.net import MobileNetV1_0_5
from model.net import MobileNetV1_1_0
from model.net import FPN as FPN
from model.net import PANet2 as PANet2
from model.net import BiFPN as BiFPN
from model.net import CBAM as CBAM
from BBLiteV4 import BBLiteV4 as BBLiteV4
from model.net import BasicConvWithoutReLU as BasicConvWithoutReLU   
 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Lightweight Coarse Crowd Counting Network (CoarseNet)   
class CoarseNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CoarseNet, self).__init__()  
        # Define the CoarseNet model architecture  
        # Backbone: Feature Extractor
        #----------------------------------
        # MobileNetV1_0.25
        #----------------------------------
#         backbone = MobileNetV1_0_25()         # Backbone: MobileNetV1_0.25
#         checkpoint = torch.load("/Path/to/your/pretrained/weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))  
#         # from collections import OrderedDict
#         new_state_dict = OrderedDict()
#         for k, v in checkpoint['state_dict'].items():      
#             name = k[7:]  # remove module
#             new_state_dict[name] = v
        
#         # load params   
#         backbone.load_state_dict(new_state_dict)   
        
#         self.body = _utils.IntermediateLayerGetter(backbone, {'stage1': 1, 'stage2': 2, 'stage3': 3})  # MobileNetV1_0.25/0.5/1.0   
#         in_channels_list = [    # MobileNetV1_0.25
#             64,
#             128,
#             256,
#             ]  

        #-----------------------------------
        # MobileNetV1_0.5
        #-----------------------------------
#         backbone = MobileNetV1_0_5()          # Backbone: MobileNetV1_0.5
#         checkpoint = torch.load("/Path/to/your/pretrained/weights/MobileNetV1x0_5.tar", map_location=torch.device('cpu'))  
#         backbone.load_state_dict(checkpoint['state_dict'])
#         self.body = _utils.IntermediateLayerGetter(backbone, {'stage1': 1, 'stage2': 2, 'stage3': 3})  # MobileNetV1_0.25/0.5/1.0   
#         in_channels_list = [    # MobileNetV1_0.5   
#             128,
#             256,
#             512,
#             ] 

        #-----------------------------------
        # MobileNetV1_1.0
        #-----------------------------------
#         backbone = MobileNetV1_1_0()          # Backbone: MobileNetV1_1.0
#         checkpoint = torch.load("/Path/to/your/pretrained/weights/MobileNetV1.tar", map_location=torch.device('cpu'))  
#         backbone.load_state_dict(checkpoint['state_dict'])
#         self.body = _utils.IntermediateLayerGetter(backbone, {'stage1': 1, 'stage2': 2, 'stage3': 3})  # MobileNetV1_0.25/0.5/1.0  
#         in_channels_list = [    # MobileNetV1_1.0  
#             256,
#             512,
#             1024,   
#             ] 
        
        #-----------------------------------
        # MobileNetV2
        #-----------------------------------
        # backbone = models.mobilenet_v2(pretrained=True)     # Backbone: MobileNetV2 
        # self.body = _utils.IntermediateLayerGetter(backbone.features, {'6': 1, '13': 2, '18': 3})   # MobileNetV2
        # in_channels_list = [     
        #     32,
        #     96,
        #     1280,  
        #     ] 

        #-----------------------------------
        # MobileNetV3-Small
        #-----------------------------------
        # backbone = models.mobilenet_v3_small(pretrained=True)     # Backbone: MobileNetV3-Small
        # self.body = _utils.IntermediateLayerGetter(backbone.features, {'3': 1, '8': 2, '12': 3})    # MobileNetV3-Small
        # in_channels_list = [
        #     24,
        #     48,
        #     576,
        #     ]          

        #-----------------------------------
        # MobileNetV3-Large
        #-----------------------------------
        # backbone = models.mobilenet_v3_large(pretrained=True)     # Backbone: MobileNetV3-Large
        # self.body = _utils.IntermediateLayerGetter(backbone.features, {'6': 1, '12': 2, '16': 3})     # MobileNetV3-Large  
        # in_channels_list = [
        #     40,
        #     112,
        #     960,
        #     ] 
        
        #-----------------------------------------------------------------------------------------------------
        
        #self.body = _utils.IntermediateLayerGetter(backbone, {'stage2': 1, 'stage3': 2, 'stage4': 3})    
        #in_channels_list = [
            #116,
            #232,
            #464,
            #]       
        
        #-----------------------------------------------------------------------------------------------------   
        
        #-----------------------------------
        # ShuffleNetV2_x0.5
        #-----------------------------------
        # backbone = models.shufflenet_v2_x0_5(pretrained=True)     # Backbone: ShuffleNetV2_x0.5
        # self.body = _utils.IntermediateLayerGetter(backbone, {'stage2': 1, 'stage3': 2, 'conv5': 3})     # ShuffleNetV2_x0.5/1.0/1.5/2.0
        # in_channels_list = [    # ShuffleNetV2_x0.5
        #     48,
        #     96,
        #     1024,   
        #     ]  

        #-----------------------------------
        # ShuffleNetV2_x1.0
        #-----------------------------------
        # backbone = models.shufflenet_v2_x1_0(pretrained=True)     # Backbone: ShuffleNetV2_x1.0
        # self.body = _utils.IntermediateLayerGetter(backbone, {'stage2': 1, 'stage3': 2, 'conv5': 3})     # ShuffleNetV2_x0.5/1.0/1.5/2.0
        # in_channels_list = [    # ShuffleNetV2_x1.0   
        #     116,
        #     232,
        #     1024,   
        #     ]       
        
        #-----------------------------------
        # ShuffleNetV2_x1.5
        #-----------------------------------
        # backbone = models.shufflenet_v2_x1_5(pretrained=True)     # Backbone: ShuffleNetV2_x1.5
        # self.body = _utils.IntermediateLayerGetter(backbone, {'stage2': 1, 'stage3': 2, 'conv5': 3})     # ShuffleNetV2_x0.5/1.0/1.5/2.0
        # in_channels_list = [    # ShuffleNetV2_x1.5
        #     176,
        #     352,
        #     1024,
        #     ]     
        
        #-----------------------------------
        # ShuffleNetV2_x2.0
        #-----------------------------------
        # backbone = models.shufflenet_v2_x2_0(pretrained=True)     # Backbone: ShuffleNetV2_x2.0 
        # self.body = _utils.IntermediateLayerGetter(backbone, {'stage2': 1, 'stage3': 2, 'conv5': 3})     # ShuffleNetV2_x0.5/1.0/1.5/2.0
        # in_channels_list = [    # ShuffleNetV2_x2.0   
        #     244,
        #     488,
        #     2048,
        #     ]           
        
        #-----------------------------------
        # EfficientNet-B0
        #-----------------------------------
        backbone = models.efficientnet_b0(pretrained=True)        # Backbone: EfficientNet-B0 
        self.body = _utils.IntermediateLayerGetter(backbone.features, {'3': 1, '5': 2, '8': 3})     # EfficientNet-B0 
        in_channels_list = [
            40,
            112,
            1280,  
            ] 
        
        #----------------------------------
        # BBLiteV4
        #----------------------------------
#         backbone = BBLiteV4()         # Backbone: BBLiteV4
#         checkpoint = torch.load("/Path/to/your/pretrained/weights/BBLiteV4.pth.tar", map_location=torch.device('cpu'))  
       
#         # load params   
#         backbone.load_state_dict(checkpoint['state_dict'])   
        
#         self.body = _utils.IntermediateLayerGetter(backbone, {'stage1': 1, 'stage2': 2, 'stage3': 3})  # MobileNetV1_0.25/0.5/1.0   
#         in_channels_list = [    # BBLiteV4
#             64,
#             128,
#             256,
#             ] 


        # Feature Refiner Module (FRM): Convolutional Block Attention Module (CBAM) 
        self.ca1 = CBAM(in_channels=in_channels_list[0], reduction_ratio=4)
        self.ca2 = CBAM(in_channels=in_channels_list[1], reduction_ratio=4)
        self.ca3 = CBAM(in_channels=in_channels_list[2], reduction_ratio=4)   
        
        # Pyramidal Feature Fusion Module (PFFM): FPN/PANet/BiFPN  
        # self.fpn = FPN(in_channels_list, 64) 
        # self.panet = PANet2(in_channels_list, 64) 
        self.bifpn = BiFPN(in_channels_list, 64)       
        
        # Density Map Generator (DMG): Density Regressor (DR) 
        self.dmg = BasicConvWithoutReLU(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)        # For FPN/BiFPN (PANet for ShuffleNetV2_x2.0)
        # self.output = BasicConvWithoutReLU(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)       # For PANet

    def forward(self, inputs):   
        # Backbone: Feature Extractor
        out = self.body(inputs)
        input = list(out.values())     
        
        # FRM: CBAM 
        ca1 = self.ca1(input[0])
        ca2 = self.ca2(input[1])
        ca3 = self.ca3(input[2])   
        
        # PFFM: FPN/PANet/BiFPN
        pf_in = [ca1, ca2, ca3]
        # x = self.fpn(pf_in) 
        # x = self.panet(pf_in)     
        x = self.bifpn(pf_in)            
        
        # DMG: DR   
        out = self.dmg(x[0])       # For FPN/BiFPN (PANet for ShuffleNetV2_x2.0)     
        # out = self.output(x[0])    # For PANet   

        return out     

#----------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------  
# Cascaded Model
#-------------------------
class CascadedCrowdNet(nn.Module):
    def __init__(self, model1, model2):
        super(CascadedCrowdNet, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x    

