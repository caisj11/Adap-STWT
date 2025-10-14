'''
Author: Caisj
Date: 2025-09-16 09:01:00
LastEditTime: 2025-10-14 09:42:08
'''
# -*- coding: utf-8 -*-
# @Time : 2023/9/5 16:19
# @Author : Caisj
import torch


class GraphConv(torch.nn.Module):
    r"""
    Graph Convolution with self feature modeling.

    Args:
        f_in: input size.
        num_cheb_filter: output size.
        conv_type:
            gcn: :math:`AHW`,
            cheb: :math:``T_k(A)HW`.
        activation: default relu.
    """
    def __init__(self, f_in, num_cheb_filter, conv_type=None, **kwargs):
        super(GraphConv, self).__init__()
        self.K = kwargs.get('K', 3) if conv_type == 'cheb' else 1
        self.with_self = kwargs.get('with_self', True)
        self.w_conv = torch.nn.Linear(f_in * self.K, num_cheb_filter, bias=False)
        if self.with_self:
            self.w_self = torch.nn.Linear(f_in, num_cheb_filter)
        self.conv_type = conv_type
        self.activation = kwargs.get('activation', torch.relu)

    def cheb_conv(self, x, adj_mx):
        bs, num_nodes, _ = x.size()

        if adj_mx.dim() == 3:
            h = x.unsqueeze(dim=1)
            h = torch.matmul(adj_mx, h).transpose(1, 2).reshape(bs, num_nodes, -1)
        else:
            h_list = [x, torch.matmul(adj_mx, x)]
            for _ in range(2, self.K):
                h_list.append(2 * torch.matmul(adj_mx, h_list[-1]) - h_list[-2])
            h = torch.cat(h_list, dim=-1)

        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def gcn_conv(self, x, adj_mx):
        h = torch.matmul(adj_mx, x)
        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def forward(self, x, adj_mx):
        self.conv_func = self.cheb_conv if self.conv_type == 'cheb' else self.gcn_conv
        return self.conv_func(x, adj_mx)
    
