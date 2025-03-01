# models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp)
        )

    def forward(self, time):
        return self.time_block(time)

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, *args):
        return self.upsample(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)
        h, _ = self.mhsa(h, h, h)
        h = h.swapaxes(2, 1).view(B, self.channels, H, W)
        return x + h

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_fn = nn.SiLU()
        self.normlize_1 = nn.BatchNorm2d(num_features=self.in_channels)
        self.conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                                padding="same")
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)
        self.normlize_2 = nn.BatchNorm2d(num_features=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1,
                                padding="same")
        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                         stride=1)
        else:
            self.match_input = nn.Identity()
        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)
        h += self.dense_1(self.act_fn(t))[:, :, None, None]
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)
        h = h + self.match_input(x)
        h = self.attention(h)
        return h

class DiffusionModel(nn.Module):
    def __init__(self, input_channels, output_channels, base_channels, base_channels_multiples,
                 apply_attention, dropout_rate, time_multiple, d_model, num_classes):
        super().__init__()
        time_emb_dims = base_channels
        time_emb_dims_exp = time_emb_dims * time_multiple
        self.time_embeddings = SinusoidalPositionEmbeddings(
            time_emb_dims=time_emb_dims, time_emb_dims_exp=time_emb_dims_exp)
        self.first = nn.Conv2d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1,
                               padding="same")
        num_resolutions = len(base_channels_multiples)
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [base_channels]
        in_channels = base_channels
        for level in range(num_resolutions):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(2):
                block = ResnetBlock(in_channels=in_channels, out_channels=out_channels, dropout_rate=dropout_rate,
                                    time_emb_dims=time_emb_dims_exp, apply_attention=apply_attention[level])
                self.encoder_blocks.append(block)
                in_channels = out_channels
                curr_channels.append(in_channels)
            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)
        self.bottleneck_block = nn.ModuleList((ResnetBlock(in_channels=in_channels, out_channels=in_channels,
                                                           dropout_rate=dropout_rate, time_emb_dims=time_emb_dims_exp,
                                                           apply_attention=True),
                                               ResnetBlock(in_channels=in_channels, out_channels=in_channels,
                                                           dropout_rate=dropout_rate, time_emb_dims=time_emb_dims_exp,
                                                           apply_attention=False)))
        self.decoder_blocks = nn.ModuleList()
        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]
            for _ in range(3):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(in_channels=encoder_in_channels + in_channels, out_channels=out_channels,
                                    dropout_rate=dropout_rate, time_emb_dims=time_emb_dims_exp,
                                    apply_attention=apply_attention[level])
                in_channels = out_channels
                self.decoder_blocks.append(block)
            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels=in_channels))
        self.ddmamba = Mamba(d_model=d_model)
        self.final = nn.Sequential(nn.GroupNorm(num_groups=8, num_channels=in_channels), nn.SiLU(),
                                   nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=3,
                                             stride=1, padding="same"))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                        nn.Linear(output_channels, num_classes))
        self.timestep_map = nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp)

    def forward(self, x, t):
        time_emb = self.time_embeddings(t)
        time_emb = self.timestep_map(time_emb)
        h = self.first(x)
        outs = [h]
        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)
        for layer in self.bottleneck_block:
            h = layer(h, time_emb)
        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)
        batch_size, channels, height, width = h.shape
        h = h.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        h = self.ddmamba(h)
        h = h.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        h = self.final(h)
        logits = self.classifier(h)
        return logits

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a[t].float()
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x, t, noise_variance_schedule):
        noise = torch.randn_like(x)
        sqrt_alpha_bar_t = self._extract(noise_variance_schedule, t, x.shape)
        x_noisy = sqrt_alpha_bar_t * x + torch.sqrt(1 - sqrt_alpha_bar_t) * noise
        return x_noisy

    def denoise(self, x_noisy, t, noise_variance_schedule):
        time_emb = self.time_embeddings(t)
        time_emb = self.timestep_map(time_emb)
        h = self.first(x_noisy)
        outs = [h]
        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)
        for layer in self.bottleneck_block:
            h = layer(h, time_emb)
        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)
        batch_size, channels, height, width = h.shape
        h = h.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        h = self.ddmamba(h)
        h = h.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        h = self.final(h)
        return h