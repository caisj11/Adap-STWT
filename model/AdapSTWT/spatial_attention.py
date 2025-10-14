# -*- coding: utf-8 -*-
# @Time : 2023/8/15 10:06
# @Author : Caisj
import torch
from math import sqrt
import numpy as np


class TriangularCausalMask():
    def __init__(self, B, N, L, device="cpu"):
        mask_shape = [B, N, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class SpatialAttention(torch.nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = torch.nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        # 输入[B,N,T,H,D]
        B, N, T, H, E = queries.shape
        _, M, _, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("bnthe,bmthe->bthnm", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bthnm,bmthd->bnthd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SpatialAttentionLayer(torch.nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(SpatialAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = torch.nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = torch.nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = torch.nn.Linear(d_model, d_values * n_heads)
        self.out_projection = torch.nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        # [B,N,T,C]
        B, N, T, C = queries.shape
        _, M, _, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries.reshape(B, -1, C)).view(B, N, T, H, -1)
        keys = self.key_projection(keys.reshape(B, -1, C)).view(B, M, T, H, -1)
        values = self.value_projection(values.reshape(B, -1, C)).view(B, M, T, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        if self.mix:
            out = out.contiguous()
        out = out.view(B, N, T, -1)

        return self.out_projection(out), attn


class SpatialSelfAttention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(SpatialSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.conv2d = torch.nn.Conv2d(1, embed_size, 1)
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        # [B,N,T,C]
        values = self.conv2d(values.transpose(1, 3)).transpose(1, 3)
        keys = self.conv2d(keys.transpose(1, 3)).transpose(1, 3)
        query = self.conv2d(query.transpose(1, 3)).transpose(1, 3)
        B, N, T, _ = query.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(B, N, T, self.heads, self.head_dim)  # embed_size维拆成 heads×head_dim
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        query = query.reshape(B, N, T, self.heads, self.head_dim)

        values = self.values(values)  # (B, N, T, heads, head_dim)
        keys = self.keys(keys)  # (B, N, T, heads, head_dim)
        queries = self.queries(query)  # (B, N, T, heads, heads_dim)

        energy = torch.einsum("bqthd, bkthd->bqkth", [queries, keys])  # 空间self-attention
        # queries shape: (B, N, T, heads, heads_dim),
        # keys shape: (B, N, T, heads, heads_dim)
        # energy: (B, N, N, T, heads)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=1)  # 在K维做softmax，和为1
        # attention shape: (B, N, N, T, heads)

        out = torch.einsum("bqkth, bkthd->bqthd", [attention, values]).reshape(B, N, T, self.heads * self.head_dim)

        out = self.fc_out(out)

        return out


class SpatialTransformer(torch.nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion=4):
        super(SpatialTransformer, self).__init__()
        self.attention = SpatialSelfAttention(embed_size, heads)
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.norm2 = torch.nn.LayerNorm(embed_size)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_size, forward_expansion * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, value, key, query):
        # Spatial Transformer
        attention = self.attention(value, key, query)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out