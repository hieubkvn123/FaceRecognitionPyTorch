import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os
import cv2
import math
import pickle
import imutils
import numpy as np

from torch.nn import Parameter
from sklearn.preprocessing import LabelEncoder

# define the custom module
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class ArcFaceNet(nn.Module):
  def __init__(self, num_classes=10):
    super(ArcFaceNet, self).__init__()

    self.resnet50 = models.resnet50(pretrained = True)

    resnet_layers = list(self.resnet50.children())
    last_layer = resnet_layers[len(resnet_layers) - 1]
 
    self.dense1 = nn.Linear(in_features = last_layer.out_features, out_features = 1024)
    self.dense1_norm = nn.BatchNorm1d(self.dense1.out_features)
    self.dense1_relu = nn.ReLU()

    self.dense2 = nn.Linear(in_features = self.dense1.out_features, out_features = 512)
    self.dense2_norm = nn.BatchNorm1d(self.dense2.out_features)
    self.dense2_relu = nn.ReLU()

    # self.arcface_layer = ArcMarginProduct(in_features=self.dense2.out_features, out_features=num_classes)

  def forward(self, inputs):
    output = self.resnet50(inputs)
    output = self.dense1(output)
    output = self.dense1_norm(output)
    output = self.dense1_relu(output)

    output = self.dense2(output)
    output = self.dense2_norm(output)
    output = self.dense2_relu(output)
    # output = self.arcface_layer(output, label)

    return output