"""Super-resolution networks used in the study.

All models are single-channel (T1w) and operate on [0, 1] intensities.
The same RRDB trunk is shared by every learned variant so that the
comparison isolates the training objective, not the architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_init(m: nn.Module, scale: float = 0.1) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        m.weight.data.mul_(scale) if scale != 1.0 else None


class ResidualBlock(nn.Module):
    def __init__(self, nf: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class SRResNet(nn.Module):
    """Classic 16-block SRResNet (Ledig et al., 2017), x4."""

    def __init__(self, nf: int = 64, nblocks: int = 16, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, nf, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(nf) for _ in range(nblocks)])
        self.body_tail = nn.Conv2d(nf, nf, 3, 1, 1)
        up = []
        for _ in range(int(scale // 2)):
            up += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True)]
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        f = self.head(x)
        f = f + self.body_tail(self.body(f))
        return base + self.tail(self.up(f))


class DenseLayer(nn.Module):
    def __init__(self, nf_in: int, growth: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(nf_in, growth, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.act(self.conv(x))


class RDB(nn.Module):
    """Residual dense block with five densely connected 3x3 layers."""

    def __init__(self, nf: int = 64, growth: int = 32):
        super().__init__()
        self.c1 = DenseLayer(nf, growth)
        self.c2 = DenseLayer(nf + growth, growth)
        self.c3 = DenseLayer(nf + 2 * growth, growth)
        self.c4 = DenseLayer(nf + 3 * growth, growth)
        self.c5 = nn.Conv2d(nf + 4 * growth, nf, 3, 1, 1)
        self.scale = 0.2

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(torch.cat([x, x1], 1))
        x3 = self.c3(torch.cat([x, x1, x2], 1))
        x4 = self.c4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.scale


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (Wang et al., 2018)."""

    def __init__(self, nf: int = 64, growth: int = 32):
        super().__init__()
        self.rdb1 = self._rdb(nf, growth)
        self.rdb2 = self._rdb(nf, growth)
        self.rdb3 = self._rdb(nf, growth)
        self.scale = 0.2

    @staticmethod
    def _rdb(nf: int, growth: int) -> nn.Module:
        return RDB(nf, growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.scale


class RRDBNet(nn.Module):
    """Compact RRDB generator (~3.4 M parameters at 4 blocks)."""

    def __init__(self, nf: int = 64, nblocks: int = 4, growth: int = 32, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, growth) for _ in range(nblocks)])
        self.body_tail = nn.Conv2d(nf, nf, 3, 1, 1)
        up = []
        for _ in range(int(scale // 2)):
            up += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True)]
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(nf, 1, 3, 1, 1)
        self.apply(lambda m: default_init(m, 1.0))

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        f = self.head(x)
        f = f + self.body_tail(self.body(f))
        return base + self.tail(self.up(f))


class UncertaintyHead(nn.Module):
    """Predicts a per-pixel error proxy in [0, 1] from the SR feature map.

    Trained post hoc on a frozen generator with an L1 objective against
    the normalised absolute error, which is markedly more stable on MPS
    than a heteroscedastic Gaussian NLL.
    """

    def __init__(self, nf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nf, 48, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(48, 24, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(24, 1, 3, 1, 1),
        )

    def forward(self, f):
        return torch.sigmoid(self.net(f))


class SRWithUncertainty(nn.Module):
    """Generator wrapper that exposes the pre-upsampling feature map."""

    def __init__(self, generator: nn.Module, nf: int = 64):
        super().__init__()
        self.g = generator
        self.unc = UncertaintyHead(nf)

    def forward(self, x):
        g = self.g
        base = F.interpolate(x, scale_factor=g.scale, mode="bicubic", align_corners=False)
        f = g.head(x)
        f = f + g.body_tail(g.body(f))
        sr = base + g.tail(g.up(f))
        u = self.unc(f)
        u = F.interpolate(u, size=sr.shape[-2:], mode="bilinear", align_corners=False)
        return sr, u


def build_model(arch: str = "rrdb", nf: int = 64, nblocks: int = 4, scale: int = 4,
                uncertainty: bool = False) -> nn.Module:
    if arch == "srresnet":
        g = SRResNet(nf=nf, nblocks=16, scale=scale)
        g.apply(lambda m: default_init(m, 1.0))
    elif arch == "rrdb":
        g = RRDBNet(nf=nf, nblocks=nblocks, scale=scale)
    else:
        raise ValueError(arch)
    # zero-init the last layer so training starts exactly from bicubic
    nn.init.zeros_(g.tail.weight)
    nn.init.zeros_(g.tail.bias)
    return SRWithUncertainty(g, nf) if uncertainty else g


if __name__ == "__main__":
    for arch in ["srresnet", "rrdb"]:
        m = build_model(arch)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 1, 64, 64)
        y = m(x)
        print(arch, f"{n/1e6:.2f}M", tuple(y.shape))
