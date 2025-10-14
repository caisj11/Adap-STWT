'''
Author: Caisj
Date: 2025-09-16 09:01:00
LastEditTime: 2025-10-13 17:52:13
'''
# -*- coding: utf-8 -*-
# @Time : 2023/8/15 10:10
# @Author : Caisj
import torch

class PositionAttention(torch.nn.Module):
    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PositionAttention, self).__init__()
        self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.conv2d = torch.nn.Conv2d(2, in_dim, 1)

    def forward(self, x):
        x = self.conv2d(x.transpose(1, 3))
        batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)

        out = self.gamma * out + x
        out = torch.einsum('bctu, bctv->uv', [out, out])
        return out