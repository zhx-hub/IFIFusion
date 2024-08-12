#the adpative input size (224-->256), ISCA relu -->sigmoid

import torch.nn as nn
import torch
import numpy as np
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import math
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from .unet_up import *
from .deit import deit_s
from logger_ import get_root_logger
from mmcv.cnn import build_norm_layer





class Up(nn.Module):
    def __init__(self, img_size= 224, embed_dim=320, channels=320,num_conv=2, norm_cfg= dict(type='BN', requires_grad=True),
                 align_corners=False, conv3x3_conv1x1=True,num_upsampe_layer=2):
        super(Up, self).__init__()
        self.num_conv = num_conv
        self.conv3x3_conv1x1 = conv3x3_conv1x1
        self.num_upsampe_layer = num_upsampe_layer
        self.align_corners = align_corners
        self.img_size = img_size
        if self.num_conv == 2:
            if self.conv3x3_conv1x1:
                self.conv_0 = nn.Conv2d(
                    embed_dim, 256, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_0 = nn.Conv2d(embed_dim, 256, 1, 1)
            self.conv_1 = nn.Conv2d(256, channels, 1, 1)
            _,self.syncbn_fc_0 = build_norm_layer(norm_cfg, 256)      
        

    def forward(self, x):
        
        if self.num_conv == 2:
            
            if self.num_upsampe_layer == 2:
                
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x, inplace=True)
           
                x = F.interpolate(
                    x, size=x.shape[-1]//2, mode='bilinear', align_corners=self.align_corners)
             
                x = self.conv_1(x)

                x = F.interpolate(
                    x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
            elif self.num_upsampe_layer == 1:
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x, inplace=True)
                x = self.conv_1(x)
                x = F.interpolate(
                    x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        return x




class SCF(nn.Module):
    def __init__(self,channel_dim=64,h=56,w=56):
        super(SCF, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d(output_size=(1,h))
        self.avg_pool_w = nn.AdaptiveAvgPool2d(output_size=(w,1))
        self.avg_pool_c=nn.AdaptiveAvgPool1d(1)
        self.cov1x1 = nn.Conv2d(channel_dim,channel_dim,kernel_size=1,stride=1)
        self.sigmoid =nn.Sigmoid()
    def forward(self,x, H, W):
        B,N,L = x.shape #
        x1 =x.permute(0,2,1).reshape(B,L,H,W)
        ap_h = self.avg_pool_h(x1)
        ap_w = self.avg_pool_w(x1)
        ap_c = self.avg_pool_c(x).permute(0,2,1).reshape(B,1,H,W)
        _hw = (ap_w@ap_h).transpose(-2,-1)
        x = self.cov1x1((_hw+ap_c))        
        x=self.sigmoid(x)
        return x
class Net(nn.Module):#SyncBN
    def __init__(self, ckpt=None, img_size=(256,256), embed_dim=256,num_classes=10, **kwargs):
        super(Net, self).__init__(**kwargs)
        
        assert type(img_size) is tuple or len(img_size)==2
        H,W = img_size[0]//16, img_size[1]//16
        self.img_size = img_size        
        self.ckpt=ckpt
        self.encoder = deit_s(img_size=self.img_size)
        self.reshape_14 = nn.Sequential( nn.Linear(256, 256), nn.GELU())

        
 
        
        self.ISCA_0 = SCF(channel_dim=384,h=H,w=W)
        self.ISCA_1 = SCF(channel_dim=384,h=H,w=W)
        self.ISCA_2 = SCF(channel_dim=384,h=H,w=W)
        self.ISCA_3 = SCF(channel_dim=384,h=H,w=W)
        
        self.outs_7= Up(img_size=(self.img_size[0]//16, self.img_size[1]//16),embed_dim=384, channels=384,num_upsampe_layer=1)
        self.outs_14= Up(img_size=(self.img_size[0]//16, self.img_size[1]//16),embed_dim=384, channels=384,num_upsampe_layer=1)
        self.outs_28= Up(img_size=(self.img_size[0]//16, self.img_size[1]//16),embed_dim=384, channels=384,num_upsampe_layer=1)
        self.outs_56= Up(img_size=(self.img_size[0]//16, self.img_size[1]//16),embed_dim=384, channels=384,num_upsampe_layer=2)
        
        
        
        
        self.down1 = nn.Conv2d(384, 384, 1, 1)
        self.down2 = nn.Conv2d(384, 384, 1, 1)                
        self.down3 = nn.Conv2d(384, 384, 1, 1)   
                    
        
        
        
        self.conv_f1 = nn.Conv2d(768, 384, kernel_size=1)
        self.conv_f2 = nn.Conv2d(768, 384, kernel_size=1)
        self.conv_f3 = nn.Conv2d(768, 384, kernel_size=1)
        self.conv_f4 = nn.Conv2d(768, 384, kernel_size=1)        

        
        self.head = nn.Linear(int(384*4),256)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head_cls = nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()
    def init_weights(self):
            
            logger = get_root_logger()
            if self.ckpt is not None:
                logger.warn(f'Load pre-trained model for '
                            f'{self.__class__.__name__} from original repo')
                for m in self.modules():

                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.weight, 1.0)
                    elif isinstance(m, nn.Conv2d):

                        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        fan_out //= m.groups
                        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                        if m.bias is not None:
                            m.bias.data.zero_()

                if True:
                    ckpt = torch.load(self.ckpt, map_location='cpu')
                    msg = self.encoder.load_state_dict(ckpt["model"], strict=False)

                        
            else:
                logger.warn(f'No pre-trained weights for '
                            f'{self.__class__.__name__}, '
                            f'training start from scratch')
                for m in self.modules():

               
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.weight, 1.0)
                    elif isinstance(m, nn.Conv2d):
                        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        fan_out //= m.groups
                        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                        if m.bias is not None:
                            m.bias.data.zero_()

        
        
        
    def forward(self, x):
        B,C,H,W = x.shape
        out_h,out_w = H//16,W//16

        out_7r, out_14r, out_28r, out_56r,patch_56r = self.encoder(x)[::-1]

        # ----------------------------SCF
 
        s1 = self.ISCA_0(patch_56r,H//16,W//16)

        patch_28r=self.down1(patch_56r.permute(0,1,2).reshape(B,384,H//16,W//16))
        s2 = self.ISCA_1(patch_28r.reshape(B,384,-1).permute(0,2,1),H//16,W//16)
        
        patch_14r=self.down2(patch_28r)       
        s3 = self.ISCA_2(patch_14r.reshape(B,384,-1).permute(0,2,1),H//16,W//16)

        patch_7r=self.down3(patch_14r)       
        s4 = self.ISCA_3(patch_7r.reshape(B,384,-1).permute(0,2,1),H//16,W//16)

        #-------------------------fusion
        f1 = self.conv_f1(torch.cat((out_7r,s4),dim=1))
        f2 = self.conv_f2(torch.cat((out_14r,s3),dim=1))
        f3 = self.conv_f3(torch.cat((out_28r,s2),dim=1))
        f4 = self.conv_f4(torch.cat((out_56r,s1),dim=1))
        
        

        x1 = self.outs_7(f1)
        x2 = self.outs_14(f2)
        x3 = self.outs_28(f3)
        x4 = self.outs_56(f4)
        

        temp = [x1,x2,x3,x4]
        inputs = torch.cat(temp, dim=1)
        out_local= self.head(inputs.permute(0,2,3,1)).permute(0,3,1,2)
        
        #-------------------------classification
        x = self.avgpool(out_7r.reshape(B,512,-1))  
        x = torch.flatten(x, 1)
        out_cls = self.head_cls(x)

        return out_local,out_cls