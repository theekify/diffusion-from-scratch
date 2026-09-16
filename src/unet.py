import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freqs = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freqs)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.downsample1 = Downsample(base_channels, base_channels * 2)

        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.downsample2 = Downsample(base_channels * 2, base_channels * 4)

        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.upsample2 = Upsample(base_channels * 4, base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.upsample1 = Upsample(base_channels * 2, base_channels)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)

        x = self.init_conv(x)

        skip1 = self.down1(x, time_emb)
        x = self.downsample1(skip1)

        skip2 = self.down2(x, time_emb)
        x = self.downsample2(skip2)

        x = self.bottleneck(x, time_emb)

        x = self.upsample2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, time_emb)

        x = self.upsample1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up1(x, time_emb)

        return self.final_conv(x)