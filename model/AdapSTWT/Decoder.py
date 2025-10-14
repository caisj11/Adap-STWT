import torch
import torch.nn as nn
import torch.nn.functional as F

from model.AdapSTWT.Encoder import MultiConv

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        d_ff = d_ff or 2 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.temporal_conv = MultiConv(in_channels=d_model, out_channels=d_model, kernel_size=3,
                                                    dilations=[1, 2, 4], dropout=dropout)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # [B,N,T,C]
        b, n, t, c = x.size()
        x_proj = x.reshape(b*n, c, t)  # [B*N, graph_dim, T]
        x_proj = self.temporal_conv(x_proj)  # [B*N, d_model, T]
        x_proj = x_proj.transpose(1, 2).reshape(b, n, t, self.d_model)  # [B, N, T, d_model]
        x_proj = self.temporal_norm(x_proj)

        x = x_proj + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x_proj + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 3))))
        y = self.dropout(self.conv2(y).transpose(1, 3))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x