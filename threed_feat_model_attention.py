# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import models
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from pointnet.model import PointNetfeat
from pointnet.model import STN3d
from pointnet.model import STNkd
import torch.nn.functional as F
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention

# import open3d as o3d

import numpy as np

class POINTCLOUDMODEL(nn.Module):
    def __init__(self, backbone,num_classes,pretrain=True,use_gpu=True,global_feat=True, feature_transform=False):
        super(POINTCLOUDMODEL, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.linear_layer = nn.Linear(1024, 2048)
        self.sa1 = ScaledDotProductAttention(d_model=2048,d_k=512,d_v=512,h=16)
        # self.sa2 = ScaledDotProductAttention(d_model=2048, d_k=512, d_v=512, h=16)
        # self.sa3 = ScaledDotProductAttention(d_model=2048, d_k=512, d_v=512, h=16)
        #add
        # self.concat_linear_layer = nn.Linear(2048, 2048)

        if self.feature_transform:
            self.fstn = STNkd(k=64)

        self._num_classes = num_classes
        self.use_gpu = use_gpu
        self.backbone = backbone
        if backbone == 'alexnet':
            self.model=models.alexnet(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            # print('f')
            # print(self.feature_size)
            del self.model.classifier[6]
            #self.fc = nn.Linear(self.feature_size,self._num_classes)
        elif backbone == 'vgg16':
            self.model=models.vgg16(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            del self.model.classifier[6]
        elif backbone == 'vgg19':
            self.model=models.vgg19(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            del self.model.classifier[6]
        elif backbone == 'resnet50':
            self.model=models.resnet50(pretrained=pretrain)
            self.feature_size = self.model.fc.in_features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            # print('f')
            # print(self.feature_size)
            # self.linear_layer_trans = nn.Linear(2048, 4096)


    def forward(self, input):
        x = input[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.linear_layer(x)


        x1 = input[1]
        x1 = x1.transpose(0, 1)
        view_pool = []

        for v in x1:
            v = v.type(torch.cuda.FloatTensor)
            feature = self.model(v)
            # if self.backbone == 'resnet50':
            #     feature = torch.flatten(feature,start_dim = 1)
            #     feature = self.linear_layer_trans(feature)
            # print(feature.shape)
            if self.backbone == 'alexnet':
                feature = feature.view(feature.size(0), self.feature_size)
                feature = feature.unsqueeze(0)
            else:
                feature = feature.view(feature.size(0), self.feature_size)  #
                feature = feature.unsqueeze(0)
            view_pool.append(feature)

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.cat((pooled_view, view_pool[i]), dim=0)  #
        # print(pooled_view.shape)
        # pooled_view = torch.mean(pooled_view, dim=0)  #
        # pooled_view = self.linear_layer_view(pooled_view)
        pooled_view = pooled_view.permute(1,0,2)
        # print(pooled_view.shape)
        # torch.Size([2, 2048])
        # torch.Size([2, 2048])
        # print(pooled_view.shape)
        # print(x.shape)
        expanded_x = x.unsqueeze(1)
        # print(expanded_x.shape)

        # expanded_pooled_view = pooled_view.unsqueeze(1)
        expanded_pooled_view = pooled_view

        ###
        # attention_input = torch.cat((expanded_x,expanded_pooled_view),dim=1)
        attention_input = expanded_pooled_view
        # print(attention_input.shape)

        attention_output1 = self.sa1(attention_input,attention_input,attention_input)
        # attention_output2 = self.sa2(attention_output1, attention_output1, attention_output1)
        # attention_output = self.sa3(attention_output2, attention_output2, attention_output2)
        output = attention_output1[:,0,:]

        # print('out')
        # print(output.shape)


        #concat = torch.cat((x, pooled_view), dim=1)
        # concat = x + pooled_view
        #
        # tmp = self.concat_linear_layer(concat)
        # tmp = torch.sigmoid(tmp)
        #print(tmp.shape)



        # a = self.concat_linear_layer(concat)
        # a = torch.sigmoid(a)
        #
        # b = self.concat_linear_layer(concat)[1]
        # b = torch.sigmoid(b)

        #res = torch.cat((x,pooled_view),dim=1)
        # res = a * x + (1.0 - a) * pooled_view
        # res = tmp * x + (1.0 - tmp) * pooled_view
        #print(res.shape)
        #exit(0)

        res = output
        # res = x

        if self.global_feat:
            return res, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat