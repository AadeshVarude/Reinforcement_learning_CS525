#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
# import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.num_actions=num_actions
        self.network=nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        h=w=self.conv_size(9,3,1)
        self.input_sz=int(h*w*64)
        self.qvals=nn.Sequential(
            nn.Linear(self.input_sz, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions))


    @staticmethod
    def conv_size(size,kernel_size,stride):
        s=(size-(kernel_size-1)-1) / stride + 1
        return s 



    def forward(self, x):
        x=x.to(device)
        x=self.network(x)
        x=x.view(x.size(0),-1)
        Qvals=self.qvals(x)
        return Qvals
