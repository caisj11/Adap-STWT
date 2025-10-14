# -*- coding: utf-8 -*-
# @Time : 2023/8/8 17:08
# @Author : Caisj
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.AdapSTWT.GCN import GraphConv
from model.AdapSTWT.channel_attention import ChannelAttention
from model.AdapSTWT.spatial_attention import SpatialTransformer
from model.AdapSTWT.tcn import TemporalConvNet


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv2d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=(3, 1),
                                  padding=(padding, 0),
                                  padding_mode='circular')
        self.norm = nn.BatchNorm2d(c_in)
        self.activation = nn.ELU()
        # self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))

    def forward(self, x):
        # x:[B,N,T,C]
        x = self.downConv(x.transpose(1, 3))
        x = self.norm(x)
        x = self.activation(x)
        # x = self.maxPool(x)
        x = x.transpose(1, 3)
        return x


class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4], dropout=0.1):
        super(MultiConv, self).__init__()
        self.dilations = dilations
        self.num_scales = len(dilations)

        base = out_channels // self.num_scales
        rem  = out_channels - base * self.num_scales
        self.out_splits = [base + (1 if i < rem else 0) for i in range(self.num_scales)]
        assert sum(self.out_splits) == out_channels

        self.conv_layers = nn.ModuleList()
        for i, dilation in enumerate(dilations):
            padding = (kernel_size - 1) * dilation // 2
            conv = nn.Conv1d(
                in_channels,
                self.out_splits[i],
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
            self.conv_layers.append(conv)

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B*N, in_channels, T]
        return: [B*N, out_channels, T]
        """
        scale_outputs = [conv(x) for conv in self.conv_layers]  # [B*N, split_i, T]
        output = torch.cat(scale_outputs, dim=1)                 # [B*N, out_channels, T]
        output = self.bn(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, attention, graph_dim, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.graph_conv_1 = GraphConv(graph_dim, d_model // 2, with_self=False)
        self.graph_conv_2 = GraphConv(graph_dim, d_model // 2, with_self=False)
        self.conv = MultiConv(in_channels=graph_dim, out_channels=d_model, kernel_size=3,
                                                    dilations=[1, 2, 4], dropout=dropout)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.conv2 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.linear = nn.Linear(graph_dim, d_model)

    def forward(self, x, adj_mx, attn_mask=None):
        # x [B, N, T, C]
        b, n, t, c = x.size()
        x_proj = x.reshape(b*n, c, t)  # [B*N, graph_dim, T]
        # ---------- step1：spatial dependency extraction ----------
        out_1 = self.graph_conv_1(x=x.reshape(-1, n, c), adj_mx=adj_mx)
        out_2 = self.graph_conv_2(x=x.reshape(-1, n, c), adj_mx=adj_mx) # [B*T,N,C]
        gcn = torch.relu(torch.cat((out_1, out_2), dim=-1).reshape(b, n, t, -1))

        # ---------- step2：temporal dependency extraction ----------
        new_x, attn = self.attention(gcn, gcn, gcn, attn_mask=attn_mask)
        x_proj = self.conv(x_proj)  # [B*N, d_model, T]
        x_proj = x_proj.transpose(1, 2).reshape(b, n, t, new_x.size(-1))  # [B, N, T, d_model]
        x_proj = self.temporal_norm(x_proj)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([x_proj, new_x], dim=-1)))
        x = gate * new_x + (1.0 - gate) * x_proj
        
        # ---------- step3：feed forward ----------
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv2(y.transpose(1, 3))))
        y = self.dropout(self.conv3(y).transpose(1, 3))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, adj_mx, attn_mask=None):
        # x:[B, T, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, adj_mx, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, adj_mx, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, adj_mx, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

