import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.temporal_fc = nn.Sequential(
            nn.Linear(512, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x, t_emb):
        b, c, _, _ = x.size()
        channel_weights = self.temporal_fc(t_emb).view(b, c, 1, 1)
        return x * channel_weights

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_fc = nn.Linear(time_emb_dims, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        if apply_attention:
            self.attention = ChannelAttention(out_channels)
            self.use_attention = True
        else:
            self.attention = nn.Identity()
            self.use_attention = False

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h += self.time_fc(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        if self.use_attention:
            h = self.attention(h, t_emb)
        else:
            h = self.attention(h)

        return h + self.shortcut(x)