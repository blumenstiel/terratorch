# reference torchgeo https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/deo.py
"""DEO (Wolf et al., CVPR 2026, https://arxiv.org/abs/2602.19863) Swin-B backbone from torchgeo.

The model has two stems, picked by the channel count of the input:

- 10 Sentinel-2 bands in the order B04, B03, B02, B05, B06, B07, B08, B8A, B11, B12
- 3 RGB bands (B04, B03, B02)

Pretraining normalization (`DEO_Weights.DEO_SWIN.transforms`, github.com/wolfilip/DEO-FM
utils/augmentations.py): the RGB bands are divided by their per-image maximum and clipped to [0, 1], then every band
is z-scored with the stats in that transform (RGB in [0, 1] units, the other bands in DN).
"""

import torch
from torch import nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

DEO_OUT_CHANNELS = [128, 256, 512, 1024]  # Swin-B stages, stride 4, 8, 16, 32


class DEOEncoderWrapper(nn.Module):
    """Returns the Swin stages in `out_indices` as [B, C, H, W] maps; the last stage goes through the final norm."""

    def __init__(self, deo_model: nn.Module, out_indices: list[int] | None = None) -> None:
        super().__init__()
        self.deo_model = deo_model
        self.out_indices = [i % 4 for i in out_indices] if out_indices else [0, 1, 2, 3]
        self.out_channels = [DEO_OUT_CHANNELS[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        assert x.shape[1] in (3, 10), f"DEO takes 10 S2 bands or RGB, got {x.shape[1]} channels"
        features = self.deo_model.forward_features(x)  # [B, H, W, C] per stage
        features[-1] = self.deo_model.feat_extr.norm(features[-1])
        return [features[i].permute(0, 3, 1, 2).contiguous() for i in self.out_indices]


@TERRATORCH_BACKBONE_REGISTRY.register
def deo_swin_b(pretrained: bool = True, out_indices: list[int] | None = None, **kwargs) -> DEOEncoderWrapper:
    """
    Args:
        pretrained (bool): Load the torchgeo `DEO_Weights.DEO_SWIN` weights.
        out_indices (list[int] | None): Swin stages to return (0-3, default all).
    Returns:
        DEOEncoderWrapper
    """
    from torchgeo.models import DEO_Weights, deo_base  # torchgeo >= 0.10.1

    return DEOEncoderWrapper(deo_base(DEO_Weights.DEO_SWIN if pretrained else None), out_indices)
