import typing as tp

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchaudio as ta
from .common import SubjectLayers, ChannelMerger
from transformers.models.longformer.modeling_longformer import LongformerLayer
from transformers.models.longformer.configuration_longformer import LongformerConfig
from transformers.models.big_bird.modeling_big_bird import BigBirdLayer, BigBirdModel 
from transformers.models.big_bird.configuration_big_bird import BigBirdConfig
from transformers.models.reformer.modeling_reformer import ReformerLayer
from transformers.models.reformer.configuration_reformer import ReformerConfig

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
                 attention_type = 'block_sparse',
                 block_size: int =  16,
                 num_random_blocks: int = 3,
                 subject_layers_dim: str = "hidden",  # or input
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
        
        self.layer_norm =  nn.LayerNorm(2 * in_channels['meg'], eps=1e-12)
        for i in range(self.depth - 1):
            
            if self.model_type == "basic":
                print('basic')
                self.layers.append(nn.TransformerEncoderLayer(d_model =in_channels['meg'],
                                                            nhead =self.nhead,
                                                            dim_feedforward=dim_ff,
                                                            batch_first=True))
            elif self.model_type == "bigbird":
                   print('bigbird')
                   self.config = BigBirdConfig(attention_type = attention_type,
                                             hidden_size =in_channels['meg'], 
                                             intermediate_size = dim_ff,
                                             num_hidden_layers =self.depth,
                                             num_attention_heads = self.nhead,
                                             num_random_blocks = num_random_blocks,
                                             block_size = block_size)
                   self.layers.append(BigBirdLayer(self.config, i))
            elif self.model_type == "reformer":
                   print('reformer')
                   self.config = ReformerConfig(attention_head_size = block_size,
                                             hidden_size =in_channels['meg'], 
                                             intermediate_size = dim_ff,
                                             num_hidden_layers =self.depth,
                                             num_attention_heads = self.nhead)
                   self.layers.append(ReformerLayer(self.config, i))
            elif self.model_type == "switchtransformer":
                   print('switchtransformer')
                   self.config = SwitchTransformersConfig(attention_head_size = block_size,
                                             hidden_size =in_channels['meg'], 
                                             intermediate_size = dim_ff,
                                             num_hidden_layers =self.depth,
                                             num_attention_heads = self.nhead)
                   self.layers.append(ReformerLayer(self.config, i))
            
                   
            elif self.model_type == "longformer":
                   print('longformer')
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
                # print('Megger')
                inputs["meg"] = self.merger(inputs["meg"], batch)       
        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)
        if self.positional_embedding is not None:
            inputs["meg"] = self.positional_embedding(inputs["meg"].permute(2, 0, 1))
            inputs["meg"] =inputs["meg"].permute(1,2,0)

        x =self.crop_or_pad(inputs['meg']).permute(0, 2, 1)
        if self.model_type == "basic":
            x =self.layers(x)
        elif self.model_type == 'reformer':
            attention_mask =torch.ones(x.size()[:-1], device=x.device)
            
            # init cached hidden states if necessary
            past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

            # concat same tensor for reversible ResNet
            # x = torch.cat([x, x], dim=-1)

            for layer in self.layers:
                x = layer(prev_attn_output=x,
                                    hidden_states=x,
                                    attention_mask = attention_mask,
                                    head_mask = None,
                                    num_hashes=None,
                                    past_buckets_states = past_buckets_states,
                                    use_cache=None,
                                    orig_sequence_length=None,
                                    output_attentions=False,
                                )
                print(x.shape)
            # # Apply layer norm to concatenated hidden states
            # x = self.layer_norm(x)
            # # Apply dropout
            # x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        elif self.model_type == "bigbird":
            attention_mask =torch.ones(x.size()[:-1], device=x.device)

            # in order to use block_sparse attention, sequence_length has to be at least
            # bigger than all global attentions: 2 * block_size
            # + sliding tokens: 3 * block_size
            # + random tokens: 2 * num_random_blocks * block_size
            max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
            if self.config.attention_type == "block_sparse" and x.size(1) <= max_tokens_to_attend:
                # change attention_type from block_sparse to original_full
                print('Switch to originall attention')
                self.config.attention_type = "original_full"

            if self.config.attention_type == "block_sparse":
                blocked_encoder_mask, band_mask, from_mask, to_mask = BigBirdModel.create_masks_for_block_sparse_attn(
                    attention_mask, self.config.block_size
                )
                extended_attention_mask = None

            elif self.config.attention_type == "original_full":
                blocked_encoder_mask = None
                band_mask = None
                from_mask = None
                to_mask = None
                # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
                # ourselves in which case we just need to make it broadcastable to all heads.
                if attention_mask.dim() == 3:
                    extended_attention_mask = attention_mask[:, None, :, :]
                elif attention_mask.dim() == 2:
                #    [batch_size, num_heads, seq_length, seq_length]
                    extended_attention_mask = attention_mask[:, None, None, :]
                extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(x.dtype).min

    

            for layer in self.layers:
                x = layer(
                        x,
                        attention_mask = extended_attention_mask,
                        head_mask = None,
                        encoder_hidden_states = None,
                        encoder_attention_mask = None,
                        band_mask = band_mask,
                        from_mask = from_mask,
                        to_mask = to_mask,
                        blocked_encoder_mask = blocked_encoder_mask,
                        past_key_value = None,
                        output_attentions = False,
                    )[0]
                
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
        
        