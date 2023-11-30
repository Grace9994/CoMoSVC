import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Mish


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        # x：[48,512,187]
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) # [48, 512, 1]
        conditioner = self.conditioner_projection(conditioner) # [48, 1024, 187]
        y = x + diffusion_step # 维度分别是 [48,512,187] & [48,512,1],将diffusion_step的最后一维复制187份和x相加

        y = self.dilated_conv(y) + conditioner # self.dilated_conv(y)形状是[48, 1024, 187]

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        # gate和filter的形状都是[48, 512, 187]
        y = torch.sigmoid(gate) * torch.tanh(filter) # [48, 512, 187]

        y = self.output_projection(y) # [48, 1024, 187]

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        #形状都是[48, 512, 187]
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
             #in_dim=vocoder_dimension 512  n_hidden: 100 n_layers: 20 n_spk: 20
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step,cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        # print("spec_shape",spec.shape)
        # print("cond_shape",cond.shape)
        cond  = cond.transpose(1,2) # [48,256,187]
        x = spec.squeeze(1)# 没有变化[48, 187, 100]
        x = x.transpose(1,2)
        x = self.input_projection(x)  # [B, residual_channel, T]
    

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        # n_layers=20，要经过20层residual layer
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers)) #[48, 512, 187]
        x = self.skip_projection(x)
        x = F.relu(x) # [48, 512, 187]
        x = self.output_projection(x)  # [B, mel_bins, T] [48, 100, 187]
        # output=x[:, None, :, :]
        # print(output.shape)
        return x[:,  :, :].transpose(1,2) # [48, 187, 100]
