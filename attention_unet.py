"""
attention_unet.py
-----------------
Attention U-Net (Oktay et al., 2018) with optional ImageNet-pretrained encoder.

Architecture:
  - Encoder: ResNet34 backbone (pretrained on ImageNet) OR vanilla conv blocks
  - Decoder: Transposed conv upsampling + skip connections
  - Attention Gates: suppress irrelevant background activations at each skip
  - Output: sigmoid-activated single-channel probability map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Double conv: (Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate (Oktay et al., 2018).

    Computes attention coefficients α ∈ [0,1] that weight the skip-connection
    features, suppressing irrelevant background regions.

    Parameters
    ----------
    F_g : channels in gating signal (from decoder)
    F_l : channels in skip connection (from encoder)
    F_int : intermediate channels
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g : gating signal from coarser scale (decoder)
        x : skip connection from encoder at same scale as output
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g1 if spatial sizes differ
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # attended skip features


class DecoderBlock(nn.Module):
    """Upsample → AttentionGate → concat → ConvBlock"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            F_g=in_ch // 2,
            F_l=skip_ch,
            F_int=skip_ch // 2,
        )
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.attention(g=x, x=skip)
        # Pad if sizes don't match exactly
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Attention U-Net (vanilla encoder)
# ---------------------------------------------------------------------------

class AttentionUNet(nn.Module):
    """
    Full Attention U-Net with vanilla (randomly initialized) encoder.
    Input:  (B, 1, H, W)  — single-channel phase-contrast image
    Output: (B, 1, H, W)  — neuron probability map (after sigmoid)

    Encoder features: [64, 128, 256, 512, 1024]
    """

    def __init__(self, in_channels: int = 1, base_features: int = 64):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,     f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder
        self.dec4 = DecoderBlock(f * 16, f * 8,  f * 8)
        self.dec3 = DecoderBlock(f * 8,  f * 4,  f * 4)
        self.dec2 = DecoderBlock(f * 4,  f * 2,  f * 2)
        self.dec1 = DecoderBlock(f * 2,  f,       f)

        # Output head
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path (with attention gates on skip connections)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Pretrained Attention U-Net (ResNet34 encoder)
# ---------------------------------------------------------------------------

class AttentionUNetPretrained(nn.Module):
    """
    Attention U-Net with ResNet34 encoder pretrained on ImageNet.
    Uses segmentation_models_pytorch for the encoder backbone.
    Transfer learning significantly reduces the data needed for convergence.

    Input:  (B, 1, H, W)  — replicated to 3 channels internally
    Output: (B, 1, H, W)  — neuron probability map
    """

    def __init__(self):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "Install segmentation_models_pytorch:\n"
                "  pip install segmentation-models-pytorch"
            )

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,       # will replicate single channel to 3
            classes=1,
            activation="sigmoid",
            decoder_attention_type="scse",  # concurrent spatial & channel SE
        )

    def forward(self, x):
        # Replicate grayscale to 3 channels (matches ImageNet expectation)
        x3 = x.repeat(1, 3, 1, 1)
        return self.model(x3)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(variant: str = "vanilla", device: str = "cpu") -> nn.Module:
    """
    Parameters
    ----------
    variant : "vanilla"    → AttentionUNet (no pretrained weights)
              "pretrained" → AttentionUNetPretrained (ResNet34 + ImageNet)
    """
    if variant == "pretrained":
        model = AttentionUNetPretrained()
    else:
        model = AttentionUNet(in_channels=1, base_features=64)

    return model.to(device)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("vanilla", device)
    x = torch.randn(2, 1, 256, 256).to(device)
    y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")   # should be (2, 1, 256, 256)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
