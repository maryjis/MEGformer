import typing as tp

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchaudio as ta
from .common import SubjectLayers, ChannelMerger
from transformers.models.longformer.modeling_longformer import LongformerLayer
from transformers.models.longformer.configuration_longformer import LongformerConfig

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
                 model_type: str = "basic",
                 attention_window: tp.Sequence[int] =[32,32,32,32],
                 # Overall structure
                 # Subject specific settings
                 subject_layers: bool = False,
                 positional_embedding: bool =False,
                 positional_embedding_dropout : float =0.0,
                 subject_dim: int = 64,
                 dim_ff: int = 2048, 
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
        self.attention_window = attention_window
        self.model_type =model_type
        self.delta = 0
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
               
        for i in range(self.depth - 1):
            
            if self.model_type == "basic":
                self.layers.append(nn.TransformerEncoderLayer(d_model =in_channels['meg'],
                                                            nhead =self.nhead,
                                                            dim_feedforward=dim_ff,
                                                            batch_first=True))
            elif self.model_type == "longformer":
                   config = LongformerConfig(attention_window = attention_window, 
                                             hidden_size =in_channels['meg'], 
                                             intermediate_size = dim_ff,
                                             num_hidden_layers =self.depth,
                                             num_attention_heads = self.nhead)
                   self.layers.append(LongformerLayer(config, i))
                   
        self.layers =nn.Sequential(*self.layers)
        self.final = nn.Linear(in_channels['meg'], out_channels)
        
    def crop_or_pad(self, x):
            length = x.size(-1)
            self.delta = self.sequence_lenth - length
            if length<self.sequence_lenth:
                return F.pad(x, (0, self.delta))
            elif length > self.sequence_lenth:
                return x[:, :, :self.sequence_lenth]
            else:
                return x
                
    def forward(self, inputs, batch):
        subjects = batch.subject_index
        length = next(iter(inputs.values())).shape[-1] 
        self.sequence_lenth = length// self.attention_window[0] * self.attention_window[0]
        
        if self.merger is not None:
                inputs["meg"] = self.merger(inputs["meg"], batch)       
        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)
        if self.positional_embedding is not None:
            inputs["meg"] = self.positional_embedding(inputs["meg"].permute(2, 0, 1))
            inputs["meg"] =inputs["meg"].permute(1,2,0)

        x =self.crop_or_pad(inputs['meg']).permute(0, 2, 1)
        if self.model_type == "basic":
            x =self.layers(x)
        elif self.model_type == "longformer":
            attention_mask =torch.zeros(x.size()[:-1])
            is_index_global_attn = attention_mask > 0
            is_global_attn = is_index_global_attn.flatten().any().item()
            for layer in self.layers:
                x = layer(
                        x,
                        attention_mask=attention_mask,
                        layer_head_mask=None,
                        is_index_masked=attention_mask < 0,
                        is_index_global_attn=is_index_global_attn,
                        is_global_attn=is_global_attn,
                        output_attentions=False,
                    )[0]    
        x =self.final(x)
        if self.delta>=0:
                return x.permute(0, 2, 1)[:, :, :length]
        else:
                return F.interpolate(x.permute(0, 2, 1), length)
        
        