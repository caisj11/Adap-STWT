# -*- coding: utf-8 -*-
# @Time : 2023/9/9 9:56
# @Author : Caisj
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.AdapSTWT.tcn import TemporalConvNet


class PositionalEmbedding(nn.Module):
    def __init__(self, graph_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, graph_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, graph_dim, 2).float() * -(math.log(10000.0) / graph_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:[B,N,T,C]
        b, n, t, c = x.size()
        pe = self.pe[:, :t, :]
        # [B,N,T,C]
        pe = pe.unsqueeze(1).expand(b, n, t, -1)
        return pe


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, graph_dim):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=graph_dim,
                                   kernel_size=(3, 1), padding=(1, 0), padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x:[B,N,T,C] --> [B*N,C,T]
        x = self.tokenConv(x.transpose(1, 3)).transpose(1, 3)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, graph_dim):
        super(TimeFeatureEmbedding, self).__init__()

        self.embed = nn.Sequential(
            nn.Linear(2, graph_dim // 2),
            nn.ReLU(),
            nn.Linear(graph_dim // 2, graph_dim)
        )

    def forward(self, x):
        # x:[B,N,T,C]
        x = self.embed(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, graph_dim, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, graph_dim=graph_dim)
        self.position_embedding = PositionalEmbedding(graph_dim=graph_dim)
        self.temporal_embedding = TimeFeatureEmbedding(graph_dim=graph_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x:[B,N,T,C]
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)