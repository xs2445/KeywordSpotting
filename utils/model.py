from enum import Enum
import hashlib
import math
import os
import random
import re

from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchaudio.transforms as T



class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class SpeechResModel(SerializableModule):
    def __init__(self, n_class, model_type="res8"):
        super().__init__()
        config = self.model_config(n_class, model_type)
        self.model_name = model_type
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        # the first convolution layer, the n_input channel = 1
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])
        # number of layers in resnet, two layers for one "block"
        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        # dilation was not used in this project
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            # y = F.sigmoid(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            # the "block" of resnet
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        x = self.output(x)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


    def model_config(self, n_class, model_type="res8"):
        """
        Type of model, should be one of:

        {res8, res15, res26, res8_narrow, res15_narrow, res26_narrow}
        """
        if model_type == "res8":
            return dict(
                n_labels=n_class, 
                n_layers=6, 
                n_feature_maps=45, 
                res_pool=(4, 3), 
                use_dilation=False)
        elif model_type == "res15":
            return dict(
                n_labels=n_class, 
                use_dilation=True, 
                n_layers=13, 
                n_feature_maps=45)
        elif model_type == "res26":
            return dict(
                n_labels=12, 
                n_layers=24, 
                n_feature_maps=45, 
                res_pool=(2, 2), 
                use_dilation=False)
        elif model_type == "res8_narrow":
            return dict(
                n_labels=12, 
                n_layers=6, 
                n_feature_maps=19, 
                res_pool=(4, 3), 
                use_dilation=False)
        elif model_type == "res15_narrow":
            return dict(
                n_labels=12, 
                use_dilation=True, 
                n_layers=13, 
                n_feature_maps=19)
        elif model_type == "res26_narrow":
            return dict(
                n_labels=12, 
                n_layers=24, 
                n_feature_maps=19, 
                res_pool=(2, 2), 
                use_dilation=False)
        else:
            raise Exception("Wrong model type, should be one of \
                    {res8, res15, res26, \
                    res8_narrow, res15_narrow, res26_narrow}")


class CNNModel(SerializableModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 2)
        self.conv3 = nn.Conv2d(16, 32, 2)
        self.conv4 = nn.Conv2d(32, 64, 2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn4(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape)
        # [batch, num_class]

        return F.log_softmax(x,dim=1) 


class NN2D(SerializableModule):
    def __init__(self, num_class):
        # super(NN2D,self).__init__()
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1)
        self.dropout1 = nn.Dropout(0.3) 
    
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.dropout2 = nn.Dropout(0.3)
        
        #self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1)
        #self.dropout3 = nn.Dropout(0.3)
        
        #self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
        #self.dropout4 = nn.Dropout(0.3)
        
        # self.fc1 = nn.Linear(384, 256)
        self.fc1 = nn.Linear(480, 256)
        self.dropout5 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256,128)
        self.dropout6 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_class)
        
    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)),kernel_size=3)
        x = self.dropout1(x)
        
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=3)
        x = self.dropout2(x)
        
        #x = F.max_pool2d(F.relu(self.conv3(x)),kernel_size=3)
        #x = self.dropout3(x)
        
        #x = F.max_pool2d(F.relu(self.conv4(x)),kernel_size=3)
        #x = self.dropout4(x)
        
        #print(x.shape)
        x = F.relu(self.fc1(x.reshape(-1,x.shape[1] * x.shape[2]*x.shape[3])))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        #print(x.shape)
        return x 