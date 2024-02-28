from functools import partial
import random
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F
import torchaudio as ta
from .simpleconv import SimpleConv
from .simpletransformer import PositionalEncoding



class CNNTransformer(SimpleConv):
    def __init__(self, *, in_channels_tranformer: int,
                 out_channels_transformer: int,
                 dim_ff: int = 2048, 
                 nhead: int =8,
                 depth_transformer: int =1, 
                 positional_embedding: bool =False,
                 positional_embedding_dropout : float =0.0,
                 **args):
        super().__init__(**args)
        self.nhead = nhead
        self.depth_transformer = depth_transformer
        self.positional_embedding =None 
        
        if positional_embedding:
            self.positional_embedding = PositionalEncoding(in_channels_tranformer, positional_embedding_dropout)
            
        self.layers =[]    
        for i in range(self.depth_transformer - 1):
    
            self.layers.append(nn.TransformerEncoderLayer(d_model =in_channels_tranformer,
                                                            nhead =self.nhead,
                                                            dim_feedforward=dim_ff,
                                                            batch_first=True))
        self.layers =nn.Sequential(*self.layers)
        self.last = nn.Linear(in_channels_tranformer, out_channels_transformer)
        
        
    def forward(self, inputs, batch):
        x =super().forward(inputs, batch)
        if self.positional_embedding is not None:
            x = self.positional_embedding(x.permute(2, 0, 1))
            x =x.permute(1,2,0)
        x =x.permute(0, 2, 1)
        x =self.layers(x)
        x =self.last(x)
        return x.permute(0, 2, 1)     
        