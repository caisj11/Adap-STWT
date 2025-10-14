# -*- coding: utf-8 -*-
# @Time : 2023/8/15 10:08
# @Author : Caisj
import torch
import config


class ChannelAttention(torch.nn.Module):
    def __init__(self, channels=64, r=4):
        super(ChannelAttention, self).__init__()
        inter_channels = int(channels // r)
        self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(1, 1))
        self.local_att = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=inter_channels, out_channels=channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=inter_channels, out_channels=channels, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.channel_fusion = torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x:[B,C,N,N]
        x = self.conv2d(x)
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        weight = self.sigmoid(xlg)
        x = x * weight
        x = self.channel_fusion(x).squeeze(1) # [B,N,N]
        x = torch.mean(x, dim=0) # [N,N]
        return x

if __name__ == "__main__":
    x = torch.rand(32, 64, 134, 134)
    CA = ChannelAttention(channels=64, r=4)
    x = CA(x)
    print(x.shape)



