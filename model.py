import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()
        half_dim = time_emb_dims // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        ts = torch.arange(total_time_steps, dtype=torch.float32)
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        self.time_block = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(time_emb_dims, time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(time_emb_dims_exp, time_emb_dims_exp)
        )

    def forward(self, time):
        return self.time_block(time)

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)

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
        h = nn.SiLU()(self.norm1(x))
        h = self.conv1(h)
        h += self.time_fc(nn.SiLU()(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = nn.SiLU()(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        if self.use_attention:
            h = self.attention(h, t_emb)
        else:
            h = self.attention(h)

        return h + self.shortcut(x)

class MultiStageClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(3):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, current_channels * 2, 3, padding=1),
                    nn.BatchNorm2d(current_channels * 2),
                    nn.ReLU(),
                    DownSample(current_channels * 2)
                )
            )
            current_channels *= 2

        self.up_layers = nn.ModuleList()
        for _ in range(3):
            self.up_layers.append(
                nn.Sequential(
                    UpSample(current_channels),
                    nn.Conv2d(current_channels, current_channels // 2, 3, padding=1),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU()
                )
            )
            current_channels = current_channels // 2

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        )

    def forward(self, x):
        skips = []
        for layer in self.down_layers:
            x = layer(x)
            skips.append(x)

        for layer, skip in zip(self.up_layers, reversed(skips)):
            x = layer(x + skip)

        return self.final(x)

class Diffusion(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, num_classes=8, d_model=32):
        super().__init__()
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dims=base_channels)
        self.ddmamba = Mamba(d_model=d_model)
        self.encoder = nn.ModuleList([
            ResnetBlock(3, 32, apply_attention=False),
            DownSample(32),
            ResnetBlock(32, 64, apply_attention=True),
            DownSample(64),
            ResnetBlock(64, 128, apply_attention=False)
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                ResnetBlock(128, 64),
                UpSample(64)
            ),
            nn.Sequential(
                ResnetBlock(64, 32),
                UpSample(32)
            ),
            nn.Sequential(
                ResnetBlock(32, 3),
                UpSample(3)
            )
        ])

        self.stage_classifiers = nn.ModuleList([
            MultiStageClassifier(in_channels=64, num_classes=num_classes),
            MultiStageClassifier(in_channels=32, num_classes=num_classes),
            MultiStageClassifier(in_channels=3, num_classes=num_classes)
        ])

        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        skips = []
        for layer in self.encoder:
            if isinstance(layer, ResnetBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
            if isinstance(layer, DownSample):
                skips.append(x)

        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        x = self.ddmamba(x)
        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        stage_outputs = []
        for decoder_stage in self.decoder:
            for layer in decoder_stage:
                if isinstance(layer, ResnetBlock):
                    x = layer(x, t_emb)
                else:
                    x = layer(x)
            stage_outputs.append(x)

        class_logits = []
        for stage_x, classifier in zip(stage_outputs, self.stage_classifiers):
            class_logits.append(classifier(stage_x))

        fused = torch.cat(class_logits, dim=1)
        return self.fusion_mlp(fused)

    def extract_features_at_steps(self, x, t_steps):
        features = []
        for t in t_steps:
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
            x = self.forward(x, t)
            features.append(x)
        return features