import typing as tp

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio as ta

class SimpleTransformer(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int],
                 out_channels: int,
                 hidden: tp.Dict[str, int],
                 n_subjects: int = 200,
                 # Overall structure
                 depth: int = 4):
        
        self.in_channels =in_channels
        self.out_channels =out_channels
        self.hidden =hidden
        self.depth =depth
        self.layers =[]
        for _ in range(self.depth - 1):
            self.layers.append(nn.TransformerEncoderLayer(d_model =in_channels['meg'],dim_feedforward=2048,batch_first=True))
        self.layers =nn.Sequential(*self.layers)
        self.final = nn.Linear(in_channels['meg'], out_channels)
        
    def forward(self, inputs, batch):
        x =inputs['meg'].permute(0, 2, 1)
        x =self.layers(inputs['meg'])
        return self.final(x)
        