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

import open3d as o3d

import numpy as np

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def read_off_file(file_path):
    with open(file_path, 'r') as file:
        # 读取 OFF 文件内容
        lines = file.readlines()

        # 提取点云信息
        num_points = int(lines[1].split()[0])
        points = [list(map(float, line.split()))[0:3] for line in lines[2:2+num_points]]

        # 如果点数少于1024，将其复制以达到1024个点
        while len(points) < 1024:
            points += points[:1024 - len(points)]

        # 如果点数多于1024，使用最远点采样
        if len(points) > 1024:
             points = farthest_point_sample(np.array(points), 1024)

        points = pc_normalize(np.array(points))
    return np.array(points)

# 读取 OFF 文件
off_file_path = '/home/sse316/heng/cgn/datasets/shrec_13/point_clouds/barn/item_1/m459.off'
point_cloud = read_off_file(off_file_path)

# 确定批次大小和点的数量
batch_size = 1  # 假设只有一个样本
num_points = point_cloud.shape[0]
print(point_cloud.shape)
print(point_cloud)
# 调整数据形状
point_cloud_input = point_cloud.reshape((batch_size, num_points, 3))
# print(type(point_cloud_input))

# 此时 point_cloud_input 可以作为 PointNet 模型的输入


# import yaml

# def load_config(file_path):
#     with open(file_path, 'r') as file:
#         try:
#             cfg = yaml.safe_load(file)
#             return cfg
#         except yaml.YAMLError as e:
#             print(f"Error loading YAML file {file_path}: {e}")
#             return None

# 例子：假设配置文件名为config.yaml
# config_file_path = 'cfgs/scanobjectnn/pointnext-s.yaml'
# config = load_config(config_file_path)

# if config:
#     # 现在你可以通过config对象访问YAML中的配置参数
#     print(config)
#     # 例如，如果YAML中有一个键为"model"，你可以这样访问它：
#     model_config = config.get('model', {})
#     print("Model configuration:", model_config)
# else:
#     print("Failed to load configuration.")


class POINTCLOUDMODEL(nn.Module):
    """definition."""

    def __init__(self, backbone,num_classes,pretrain=True,use_gpu=True):
        super(POINTCLOUDMODEL, self).__init__()
        self._num_classes = num_classes
        self.use_gpu = use_gpu
        self.backbone = backbone

        """Build pre-trained resnet34 model for feature extraction of 3d model render images
        """


        # self.model=models.alexnet(pretrained=pretrain)
        cfg = EasyConfig()
        cfg.load('/home/sse316/heng/cgn/cfgs/scanobjectnn/pointnext-s.yaml', recursive=True)
        self.model = build_model_from_cfg(cfg.model).cuda()

        # self.feature_size = self.model.classifier[6].in_features
        # del self.model.classifier[6]
        #self.fc = nn.Linear(self.feature_size,self._num_classes)


    def forward(self, x):
        """
        Args:
            x: input a batch of image

        Returns:
            pooled_view: Extracted features, maxpooling of multiple features of 12 view_images of 3D model

            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """
        # x = x.transpose(0, 1)
        # pool = []

        # print('x', x['pos'].shape,x['x'].shape)
        # v = x.type(torch.cuda.FloatTensor)
        # feature = self.model.forward(x)
        # 调整数据形状
        # point_cloud_input = x.reshape((1, 1024, 3))
        # import torch
        data = torch.tensor(x, dtype=torch.float).to('cuda:0')
        import torch.nn.functional as F
        x1 = F.pad(data, (0, 1)).to('cuda:0')
        x1 = x1[:, :, :4].transpose(1, 2).contiguous()
        pos = data.type(torch.float)
        x1 = x1.type(torch.float)
        pos.to('cuda:0')
        x1.to('cuda:0')

        inputValue = {'pos': pos, 'x': x1}
        feature = self.model.encoder.forward_cls_feat(inputValue)
        return feature

        # feature = feature.view(feature.size(0), self.feature_size)
        # feature = feature.unsqueeze(0)

        # for v in x:
        #     v = v.type(torch.cuda.FloatTensor)
        #     feature = self.model(v)
        #
        #     feature = feature.view(feature.size(0), self.feature_size)
        #     feature = feature.unsqueeze(0)
        #
        #     pool.append(feature)
        #
        # pooled_view = pool[0]
        # for i in range(1, len(pool)):
        #     pooled_view = torch.cat((pooled_view, pool[i]),dim=0)  #
        # pooled_view = torch.mean(pooled_view,dim=0)  #
        # return pooled_view

pc_model = POINTCLOUDMODEL('PointNeXt',1024)
# data = torch.tensor(point_cloud_input, dtype=torch.float).to('cuda:0')

# import torch.nn.functional as F
# x = F.pad(data, (0, 1)).to('cuda:0')
# x = x[:, :, :4].transpose(1, 2).contiguous()
# pos = data.type(torch.float)
# x = x.type(torch.float)
# pos.to('cuda:0')
# x.to('cuda:0')
#
# inputValue = {'pos':pos,'x':x}

res = pc_model.forward(point_cloud)

print(res,res.shape)