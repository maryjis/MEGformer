import typing as tp

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchaudio as ta
from .common import SubjectLayers, ChannelMerger

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class SimpleTransformer(nn.Module):
    def __init__(self,
                 # Channels
                 in_channels: tp.Dict[str, int],
                 out_channels: int,
                 hidden: tp.Dict[str, int],
                 n_subjects: int = 200,
                 # Overall structure
                 # Subject specific settings
                 subject_layers: bool = False,
                 positional_embedding: bool =False,
                 positional_embedding_dropout : float =0.0,
                 subject_dim: int = 64,
                 subject_layers_dim: str = "input",  # or hidden
                 subject_layers_id: bool = False,
                 merger: bool = False,
                 merger_pos_dim: int = 256,
                 merger_channels: int = 270,
                 merger_dropout: float = 0.2,
                 merger_penalty: float = 0.,
                 merger_per_subject: bool = False,
                 nhead: int =8,
                 depth: int = 4):
        super().__init__()
        self.in_channels =in_channels
        self.out_channels =out_channels
        print('self.in_channels', self.in_channels)
        print('self.out_channels', self.out_channels)
        self.hidden =hidden
        self.depth =depth
        self.nhead =nhead
        self.layers =[]
        self.subject_layers = None
        
        self.merger = None
  
        if merger:
            self.merger = ChannelMerger(
                merger_channels, pos_dim=merger_pos_dim, dropout=merger_dropout,
            usage_penalty=merger_penalty, n_subjects=n_subjects, per_subject=merger_per_subject)
        in_channels["meg"] = merger_channels
                
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects, subject_layers_id)
            in_channels["meg"] = dim
            
        self.positional_embedding = None    
        if positional_embedding:
            self.positional_embedding = PositionalEncoding(in_channels['meg'], positional_embedding_dropout)
               
        for _ in range(self.depth - 1):
            self.layers.append(nn.TransformerEncoderLayer(d_model =in_channels['meg'],
                                                          nhead =self.nhead,
                                                          dim_feedforward=2048,
                                                          batch_first=True))
        self.layers =nn.Sequential(*self.layers)
        self.final = nn.Linear(in_channels['meg'], out_channels)
        
    def forward(self, inputs, batch):
        subjects = batch.subject_index
        if self.merger is not None:
                inputs["meg"] = self.merger(inputs["meg"], batch)       
        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)
        if self.positional_embedding is not None:
            inputs["meg"] = self.positional_embedding(inputs["meg"].permute(2, 0, 1))
            inputs["meg"] =inputs["meg"].permute(1,2,0)
    
        x =inputs['meg'].permute(0, 2, 1)
        x =self.layers(x)
        x =self.final(x)
        return x.permute(0, 2, 1)
        