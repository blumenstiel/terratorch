# DOFA

DOFA (Dynamic One-For-All) is a multi-sensor foundation model. Its patch embedding is generated from the central
wavelength of each input band, so one set of weights takes any number of bands from any optical or SAR sensor.
TerraTorch wraps the [torchgeo implementation](https://torchgeo.readthedocs.io/en/stable/api/models.html#dofa).
Paper: [Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation](https://arxiv.org/abs/2403.15356),
code: [zhu-xlab/DOFA](https://github.com/zhu-xlab/DOFA).

---

## Model versions

```text
dofa_small_patch16_224   # no pretrained weights
dofa_base_patch16_224    # DOFA MAE weights (torchgeo DOFABase16_Weights.DOFA_MAE)
dofa_large_patch16_224   # DOFA MAE weights (torchgeo DOFALarge16_Weights.DOFA_MAE)
```

All are ViTs with patch size 16, pretrained at 224 x 224 px.

## Bands and wavelengths

`model_bands` lists the input channels in order. Each name is mapped to a wavelength (in µm) that conditions the
patch embedding, so the band names must match the data:

| Band | Wavelength | Band | Wavelength |
|---|---|---|---|
| `COASTAL_AEROSOL` | 0.44 | `NIR_NARROW` | 0.864 |
| `BLUE` | 0.49 | `WATER_VAPOR` | 0.945 |
| `GREEN` | 0.56 | `CIRRUS` | 1.373 |
| `RED` | 0.665 | `SWIR_1` | 1.61 |
| `RED_EDGE_1` | 0.705 | `SWIR_2` | 2.20 |
| `RED_EDGE_2` | 0.74 | `THERMAL_INFRARED_1` | 10.90 |
| `RED_EDGE_3` | 0.783 | `THERMAL_INFRARED_12` | 12.00 |
| `NIR_BROAD` | 0.832 | `VV`, `VH` (and `ASC_*`, `DSC_*`) | 3.75 |

**SAR:** DOFA v1 was pretrained with Sentinel-1 using the placeholder wavelength 3.75 for both channels (DOFA
`pretraining/datasets/waves.json`), not the physical C-band wavelength, so `VV`/`VH` map to 3.75. Since both channels
share one wavelength, they also share one generated kernel.

Every DOFA pretraining batch came from a single sensor. Feed optical and SAR bands in separate forward passes rather
than as one stacked tensor: the weight generator attends over all input wavelengths together, so adding SAR
wavelengths would also change the optical kernels, a combination the model never saw.

## Input size

The checkpoints have positional embeddings for 224 x 224 px. For other sizes, pass `img_size`; the pretrained
positional embedding is then interpolated to the new grid (`pos_interpolation_mode`, default `bilinear`) when the
weights are loaded.

```python
from terratorch.registry import BACKBONE_REGISTRY

model = BACKBONE_REGISTRY.build(
    "dofa_base_patch16_224",
    pretrained=True,
    model_bands=["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"],
    img_size=256,
)
```

Pretrained weights are loaded from torchgeo (`weights`, default) or from a checkpoint file or Hugging Face URL
(`ckpt_data`, e.g. `https://hf.co/torchgeo/dofa/resolve/<revision>/dofa_base_patch16_224-a0275954.pth`; the
revision in the URL is respected).

## Normalization

The model does not normalize its input. DOFA's pretraining z-scored each band per dataset. For Sentinel-2, it saw
SatlasPretrain L1C bands B02, B03, B04 (TCI, 0–2000 DN mapped to 0–255), B05, B06, B07, B08, B11 and B12 (DN / 32).
In DN, the corresponding mean / std per band are:

| Band | Mean | Std | Band | Mean | Std |
|---|---|---|---|---|---|
| B02 | 993 | 529 | B07 | 3325 | 1962 |
| B03 | 900 | 549 | B08 | 3245 | 1930 |
| B04 | 895 | 610 | B11 | 2314 | 1533 |
| B05 | 2698 | 2067 | B12 | 1814 | 1363 |
| B06 | 3130 | 1974 | | | |

B01, B8A and B09 were not part of its Sentinel-2 data.

## Output

The backbone returns a list with the tokens of each block in `out_indices` (default: all blocks), shape
`[B, 1 + (H / 16) * (W / 16), D]` with the CLS token first. Use a neck such as `SelectIndices` +
`ReshapeTokensToImage` for dense tasks.
