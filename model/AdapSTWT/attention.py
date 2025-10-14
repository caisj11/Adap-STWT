# -*- coding: utf-8 -*-
# @Time : 2023/9/1 9:45
# @Author : Caisj
import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, N, L, device="cpu"):
        mask_shape = [B, N, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class Attention(torch.nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(Attention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = torch.nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # 输入[B,N,T,H,D]
        B, N, L, H, E = queries.shape
        _, _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("bnlhe,bnshe->bnhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, N, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bnhls,bnshd->bnlhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        # [B,N,T,C]
        B, N, L, C = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries.reshape(B, -1, C)).view(B, N, L, H, -1)
        keys = self.key_projection(keys.reshape(B, -1, C)).view(B, N, S, H, -1)
        values = self.value_projection(values.reshape(B, -1, C)).view(B, N, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, N, L, -1)

        return self.out_projection(out), attn