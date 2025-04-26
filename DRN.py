import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import Conv_4_copy,ResNet_copy
import math
from utils.l2_norm import l2_norm
from models.modules.CLFR import CLFR
from models.modules.FSRM import FSRM
from kan import kan

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class DRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False, args=None):
        
        super().__init__()
        self.args = args
        self.short_cut_weight =self.args.short_cut_weight
        self.disturb_num = args.disturb_num
        self.resnet = resnet 
        self.resolution = 5*5

        if self.resnet:
            self.num_channel = 640
            self.num_channel2 = 640
            self.feature_size = 640
            self.feature_extractor = ResNet_copy.resnet12(drop=True)
            self.dim = self.num_channel*5*5

            self.conv_block3 = nn.Sequential(
                BasicConv(self.num_channel // 2, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.conv_block4 = nn.Sequential(
                BasicConv(self.num_channel, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.max3 = nn.AdaptiveMaxPool2d((1, 1))
            self.max4 = nn.AdaptiveMaxPool2d((1, 1))

            self.both_mlp2 = nn.Sequential(
                nn.BatchNorm1d(self.num_channel2 * self.disturb_num),
                nn.Linear(self.num_channel2 * self.disturb_num, self.num_channel2 * self.disturb_num),
                nn.ELU(inplace=True)
            )
            self.both_mlp3 = nn.Sequential(
                nn.BatchNorm1d(self.feature_size),
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.both_mlp4 = nn.Sequential(
                nn.BatchNorm1d(self.feature_size),
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.mask_branch = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, self.disturb_num, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.num_channel = 64
            self.num_channel2 = 64
            self.feature_size = 64 *5 *5
            self.feature_extractor = Conv_4_copy.BackBone(self.num_channel)            
            self.dim = self.num_channel*5*5
            self.avg = nn.AdaptiveAvgPool2d((5, 5))
            self.both_mlp2 = nn.Sequential(
                nn.Linear(self.num_channel2 * self.disturb_num, self.num_channel2 * self.disturb_num),
                nn.ELU(inplace=True)
            )
            self.both_mlp3 = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.both_mlp4 = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
            self.mask_branch = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, self.disturb_num, kernel_size=1, stride=1, padding=0)
            )
                
        kan_c = self.num_channel
        self.kan1 = kan(in_features=kan_c, hidden_features=kan_c, act_layer=nn.GELU, drop=0., version=4)
        self.kan2 = kan(in_features=kan_c, hidden_features=kan_c, act_layer=nn.GELU, drop=0., version=4)

        self.clfr = CLFR(self.resnet, self.num_channel)
        self.fsrm1 = FSRM(
                sequence_length=self.resolution,
                embedding_dim=self.num_channel,
                num_layers=1,
                num_heads=1,
                mlp_dropout_rate=0.,
                attention_dropout=0.,
                positional_embedding='sine'
                )
        self.d = self.num_channel
        self.shots = shots
        self.way = way
        self.resnet = resnet
        # #缩放参数
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        # #W（用于加权相似度的权重参数）
        self.W = nn.Parameter(torch.full((2, 1), 1. / 2), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    #接收输入张量 inp，通过特征提取器获取特征图。
    #将特征图输入到 FSRM 模型中，获取特征向量
    def get_feature_vector(self,inp):
        f_h, f_m, f_n= self.feature_extractor(inp)#fh 200, 640, 5, 5 l3 200, 320, 10, 10

        return f_h, f_m, f_n#f_h 第四层 f_m第三层 f_n第二层
    
    def integration(self, layer1, layer2):

        batch_size = layer1.size(0)
        channel_num = layer1.size(1)
        disturb_num = layer2.size(1)
        layer1 = layer1.unsqueeze(2)
        layer2 = layer2.unsqueeze(1)

        sum_of_weight = layer2.view(batch_size, disturb_num, -1).sum(-1) + 0.00001
        vec = (layer1 * layer2).view(batch_size, channel_num, disturb_num, -1).sum(-1)
        vec = vec / sum_of_weight.unsqueeze(1)
        vec = vec.view(batch_size, channel_num*disturb_num)
        return vec
        
    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False):
        
        batch_size = inp.size(0)
        #f_h 第四层 f_m第三层
        f_h, f_m, f_n = self.get_feature_vector(inp)
       
        f_refine_nm, f_refine_mh, f_refine_h= self.clfr(f_h, f_m, f_n)#400 64 5 5
       
        f_refine_h = self.fsrm1(f_refine_h).transpose(1, 2).view(batch_size, self.num_channel, 5, 5).contiguous()#400 64 5 5

        ##f_h = self.fsrm1(f_h).transpose(1, 2).view(batch_size, self.num_channel, 5, 5).contiguous()#400 64 5 5
        
        B, C, H, W = f_refine_h.shape
        x = f_refine_h.reshape(B,C, H*W).permute(0, 2, 1)
        x = self.kan1(x, H, W)
        #x = self.kan2(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        f_refine_h = x.contiguous()

        #f_refine_h = self.fsrm1(f_refine_h).transpose(1, 2).view(batch_size, self.num_channel, 5, 5).contiguous()#400 64 5 5


        #4融2 mask改成这个名称-----Dynamic Feature Mask
        heat_map = self.mask_branch(f_refine_h)
        mask = nn.Sigmoid()(heat_map)
        layernm_vec0 = self.integration(f_refine_nm, mask) #conv4 [400, 320]

        layernm_vec = (1 - self.short_cut_weight) * self.both_mlp2(layernm_vec0) + self.short_cut_weight * layernm_vec0 #conv4 [400, 320]
        support_f3 = layernm_vec[:way * shot].view(way, shot, self.num_channel2 * self.disturb_num).mean(1)#conv4 [20, 320]
        query_f3 = layernm_vec[way * shot:] #Conv4 [300, 320]
        cos_f3 = F.linear(l2_norm(query_f3), l2_norm(support_f3))
        
        #4融3
        heat_map1 = self.mask_branch(f_refine_h)
        mask = nn.Sigmoid()(heat_map1)
        layermh_vec0 = self.integration(f_refine_mh, mask)

        layermh_vec = (1 - self.short_cut_weight) * self.both_mlp2(layermh_vec0) + self.short_cut_weight * layermh_vec0 #[400, 320]
        support_f4 = layermh_vec[:way * shot].view(way, shot, self.num_channel2 * self.disturb_num).mean(1)
        query_f4 = layermh_vec[way * shot:]
        cos_f4 = F.linear(l2_norm(query_f4), l2_norm(support_f4))

        if self.resnet:#200 640 5 5
            f_mm = self.conv_block3(f_m)
            f_hh = self.conv_block4(f_h)
            f_mm = self.max3(f_mm)
            f_mm = f_mm.view(f_mm.size(0), -1)
            f_hh = self.max4(f_hh)
            f_hh = f_hh.view(f_hh.size(0), -1)
        else:
            f_mm = self.avg(f_m)
            f_mm = f_mm.view(f_mm.size(0), -1)
            f_hh = f_h.view(f_h.size(0), -1)


        f_hh = (1 - self.short_cut_weight) * self.both_mlp4(f_hh) + self.short_cut_weight * f_hh
        support_f5 = f_hh[:way * shot].view(way, shot, -1).mean(1)
        query_f5 = f_hh[way * shot:]
        cos_f5 = F.linear(l2_norm(query_f5), l2_norm(support_f5))

        f_mm = (1 - self.short_cut_weight) * self.both_mlp3(f_mm) + self.short_cut_weight * f_mm
        support_f6 = f_mm[:way * shot].view(way, shot, -1).mean(1)
        query_f6 = f_mm[way * shot:]
        cos_f6 = F.linear(l2_norm(query_f6), l2_norm(support_f6))


        return cos_f3, cos_f4, cos_f5, cos_f6

    def meta_test(self,inp,way,shot,query_shot):

        cos_f3, cos_f4, cos_f5, cos_f6 = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)
        scores = cos_f3 + cos_f4 + cos_f5 + cos_f6 
        _,max_index = torch.max(scores,1)

        return max_index

    def forward(self, inp):

        cos_f3,cos_f4, cos_f5, cos_f6 = self.get_neg_l2_dist(inp=inp, way=self.way, shot=self.shots[0],query_shot=self.shots[1])

        return cos_f3,cos_f4,cos_f5, cos_f6
