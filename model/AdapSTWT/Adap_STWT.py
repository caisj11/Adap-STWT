# -*- coding: utf-8 -*-
# @Time : 2023/7/13 21:11
# @Author : Caisj


import torch
import torch.nn as nn
from model.AdapSTWT.Decoder import Decoder, DecoderLayer
from model.AdapSTWT.Encoder import Encoder, ConvLayer, EncoderLayer
from model.AdapSTWT.attention import AttentionLayer, Attention
from model.AdapSTWT.embed import DataEmbedding


class Models(torch.nn.Module):
    """
        encoder_num: The number of Block.
        node_num: The number of nodes.
        time_step: input time step.
        conv_dim: hidden size of Conv2d.
        graph_dim: Graph embedding dimension.
        embed_size: Transformer embedding dimension.
        head: Transformer heads number
        dropout: dropout number
        forward_expansion: Magnification of the embedded layer in Transformer
    """

    def __init__(self, e_layers, d_layers, node_num, time_step, pre_num, graph_dim,
                 d_model, dropout, device, heads=4, d_ff=256, activation='gelu',
                 distil=True, mix=True, output_attention=False):
        super(Models, self).__init__()
        self.device = device
        self.node_num = node_num
        self.time_step = time_step
        self.pre_num = pre_num
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(2, graph_dim, dropout)
        self.dec_embedding = DataEmbedding(1, graph_dim, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attention(False, attention_dropout=dropout), d_model, heads),
                    graph_dim,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attention(True, attention_dropout=dropout, output_attention=False),
                                   d_model, heads, mix=mix),
                    AttentionLayer(Attention(False, attention_dropout=dropout, output_attention=False),
                                   d_model, heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, adj_mx, enc_self_mask=None, dec_self_mask=None, cross_mask=None):
        # [b,t,n,c] --> [b,n,t,c]
        x_enc, x_mark_enc = x_enc.transpose(1, 2), x_mark_enc.transpose(1, 2)
        x_dec, x_mark_dec = x_dec.transpose(1, 2), x_mark_dec.transpose(1, 2)
        # temporal embedding, positional encoding
        enc_in = self.enc_embedding(x_enc, x_mark_enc)
        dec_in = self.dec_embedding(x_dec, x_mark_dec)
        # Encode
        enc_out, attns = self.encoder(enc_in, adj_mx, attn_mask=enc_self_mask)
        # Decode
        dec_out = self.decoder(dec_in, enc_out, x_mask=dec_self_mask, cross_mask=cross_mask)
        dec_out = self.projection(dec_out).squeeze(-1).transpose(1, 2) # [B,T,N]

        if self.output_attention:
            return dec_out[:, -self.pre_num:, :], attns
        else:
            return dec_out[:, -self.pre_num:, :]  # [B, T, N]