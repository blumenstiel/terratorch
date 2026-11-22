# Clay v1.5

[Clay v1.5](https://github.com/Clay-foundation/model) is a ViT-L MAE (patch 8) with a wavelength-conditioned patch
embedding, so one encoder accepts any sensor whose band wavelengths are known.

## Backbone

```python
from terratorch.registry import BACKBONE_REGISTRY

backbone = BACKBONE_REGISTRY.build(
    "clay_v1_5_large",
    pretrained=True,              # made-with-clay/Clay v1.5/clay-v1.5.ckpt, or pass checkpoint_path
    platform="sentinel-2-l2a",    # key in the metadata
    out_indices=[5, 11, 17, 23],  # default [-1]
)
features = backbone(x)  # list of [B, 1024, H / 8, W / 8]
```

Loading fails if the weights cannot be downloaded or the checkpoint does not match the encoder.

The last block is returned after the final LayerNorm, intermediate blocks without it.
`H` and `W` need not be equal.

## Metadata and normalization

`terratorch.models.backbones.clay_v15.metadata.CLAY_METADATA` holds Clay's per-sensor GSD and band wavelength,
mean and std (`sentinel-2-l2a`, `sentinel-1-rtc`, `landsat-c2l1`, `landsat-c2l2-sr`, `naip`, `linz`,
`planetscope-sr`, `modis`, `satellogic-MSI-L1D`). The input channels must follow that platform's `band_order`.
For another sensor or band subset, pass your own `metadata` with the same layout. Clay was pretrained on
`(x - mean) / std` with these statistics.

## Time and location

Clay encodes acquisition time and location as 8 extra channels of the position encoding. Pass them as forward
kwargs, each `[B, 4]`, in Clay's normalization: `time = [sin(week), cos(week), sin(hour), cos(hour)]` and
`latlon = [sin(lat), cos(lat), sin(lon), cos(lon)]`, with ISO week `* 2 pi / 52`, hour `* 2 pi / 24` and
lat / lon in radians.
Both default to zero, as in the 20% of pretraining samples where they were dropped.

```python
features = backbone(x, time=time, latlon=latlon)
```

## Model factory

`Clay1_5ModelFactory` builds a full segmentation, regression or classification model on the same backbone. It
defaults to the ViT-L size and `CLAY_METADATA`; `platform` is required and `out_indices` selects the decoder inputs.
