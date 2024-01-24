import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.nn.utils import weight_norm
import math
import typing as tp
from .common import SubjectLayers, ChannelMerger

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
    
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self,
                 sequence_lenth,
                 d_model,
                 d_ff,
                 num_kernels,
                 top_k):
        super(TimesBlock, self).__init__()
        self.seq_len = sequence_lenth
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len// period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res
    
    
class TimesNet(nn.Module):
    def __init__(self, # Channels
                 in_channels: tp.Dict[str, int],
                 out_channels: int,
                 hidden: tp.Dict[str, int],
                 n_subjects: int = 200,
                 # Overall structure
                 # Subject specific settings
                 subject_layers: bool = False,
                 subject_dim: int = 64,
                 subject_layers_dim: str = "input",  # or hidden
                 subject_layers_id: bool = False,
                 # Attention multi-dataset support
                 merger: bool = False,
                 merger_pos_dim: int = 256,
                 merger_channels: int = 270,
                 merger_dropout: float = 0.2,
                 merger_penalty: float = 0.,
                 merger_per_subject: bool = False,
                 sequence_lenth: int = 361,   
                 num_kernels: int = 6,
                 top_k: int = 10,
                 dropout_projection: float = 0.2,
                 d_model: int = 32, 
                 d_ff: int = 32 ,
                 flatten_out_channels: int =1024,
                 depth: int =2
                 ):
            super().__init__()
            self.sequence_lenth = sequence_lenth
            self.delta = 0
            self.depth = depth            
            self.merger = None
  
            if merger:
                self.merger = ChannelMerger(
                    merger_channels, pos_dim=merger_pos_dim, dropout=merger_dropout,
                    usage_penalty=merger_penalty, n_subjects=n_subjects, per_subject=merger_per_subject)
                in_channels["meg"] = merger_channels
            
            self.subject_layers =None
            
            if subject_layers:
                assert "meg" in in_channels
                meg_dim = in_channels["meg"]
                dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
                self.subject_layers = SubjectLayers(meg_dim, dim, n_subjects, subject_layers_id)
                in_channels["meg"] = dim
                    
            self.enc_embedding = DataEmbedding(in_channels["meg"], d_model)
            
            self.model = nn.ModuleList([TimesBlock(sequence_lenth,
                                                    d_model,
                                                    d_ff,
                                                    num_kernels,
                                                    top_k)
                                    for _ in range(depth)])
            self.layer_norm = nn.LayerNorm(d_model)
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout_projection)
            self.projection = nn.Linear(d_model, flatten_out_channels)
        
    def crop_or_pad(self, x):
            length = x.size(-1)
            self.delta = self.sequence_lenth - length
            if length<self.sequence_lenth:
                return F.pad(x, (0, self.delta))
            elif length > self.sequence_lenth:
                return x[:, :, :self.sequence_lenth]
            else:
                return x
        
    def forward(self,inputs, batch):
        
            subjects = batch.subject_index
            length = next(iter(inputs.values())).shape[-1]  # length of any of the inputs
            #subject layer  
            if self.merger is not None:
                inputs["meg"] = self.merger(inputs["meg"], batch)
                
            if self.subject_layers is not None:
                inputs["meg"] = self.subject_layers(inputs["meg"], subjects)

            x =self.crop_or_pad(inputs['meg'])
            x =x.permute(0, 2, 1)  
            # embedding
            if self.enc_embedding is not None:
                x= self.enc_embedding(x, None)  # [B,T,C]
                   
            # TimesNet
            for i in range(self.depth):
                enc_out = self.layer_norm(self.model[i](x))

            # Output
            # the output transformer encoder/decoder embeddings don't include non-linearity
            output = self.act(enc_out)
            output = self.dropout(output)
            # project back  #[B,T,d_model]-->[B,T,c_out]
            output = self.projection(output) 
            output = output.permute(0, 2, 1)
            if self.delta>=0:
                return output[:, :, :length]
            else:
                return F.interpolate(output, length)[0] 
