# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import timm


# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# !huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir content/lag-llama




class SketchModel(nn.Module):
    """ definition."""

    def __init__(self,backbone, num_classes,pretrain = True,use_gpu=True):
        super(SketchModel, self).__init__()
        self._num_classes = num_classes
        self.use_gpu = use_gpu
        self.backbone = backbone

        """Build pre-trained resnet101 model for feature extraction of sketch images
        """
        if backbone == 'alexnet':
            self.model=models.alexnet(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            del self.model.classifier[6]
            self.fc = nn.Linear(self.feature_size,self._num_classes)
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
            # del self.model.fc
        elif backbone == 'vitb':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrain, pretrained_cfg_overlay=dict(file='/home/sse316/heng/cgn/vitb/pytorch_model.bin'))
            # 假设您的 ViT 模型的分类器是最后一个全连接层
            # eself.feature_size = self.model.head.in_features
            # self.model.had = nn.Linear(self.feature_size, self._num_classes)
            self.feature_size = self.model.head.in_features  # 获取 ViT 模型的输出特征大小
            self.model.head = nn.Identity()  # 移除最后的分类头，仅保留特征提取器

    def forward(self, x):
        """
        Args:
            x: input a batch of image

        Returns:
            feature: Extracted features,feature matrix with shape (batch_size, feat_dim),which to be passed
                to the Center Loss

            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """

        if self.backbone == 'alexnet':
            feature = self.model(x)
            feature = feature.view(-1, self.feature_size)
        elif self.backbone == 'vitb':
            # ViT 模型的 forward 方法可能不同，具体取决于您使用的库
            # 这里假设 timm 库的 ViT 模型返回的是一个包含特征和分类结果的元组
            feature = self.model(x)
            # feature = outputs[:, 1:].reshape(-1, self.feature_size)
            # feature, logits = outputs[0], outputs[1]
            # print(outputs.shape,222)
            # return feature, logits
        else:
            feature = self.model(x)
            # print(feature.shape,33)
            feature = feature.view(-1, self.feature_size)

        #feature = self.layer1(feature)
        #feature = self.layer2(feature)
        #feature = self.layer3(feature)
        #logits = self.fc(feature)

        return feature




