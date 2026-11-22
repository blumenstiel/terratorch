import logging
from collections.abc import Callable
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch import nn
from box import Box

import terratorch.models.decoders as decoder_registry
from terratorch.models.backbones.clay_v15.metadata import CLAY_METADATA
from terratorch.models.backbones.clay_v15.model import Encoder
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import DecoderNotFoundError, extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY, TERRATORCH_BACKBONE_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification", "scalar_regression"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS

# ClayMAE checkpoint keys that collide with the terratorch decoder_ prefix convention.
# These are ignored when building the terratorch decoder kwargs.
_CLAY_DECODER_KEYS = {"dim", "depth", "heads", "dim_head", "mlp_ratio", "ratio"}

# Encoder of the released Clay v1.5 checkpoint (ViT-L, patch 8).
CLAY_V1_5_LARGE = dict(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0, patch_size=8)


def _strip_clay_keys(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _CLAY_DECODER_KEYS}


logger = logging.getLogger("terratorch")


class ClayMAEBackbone(nn.Module):
    """Wraps the Clay v1.5 Encoder as a feature extractor for downstream tasks.

    Runs the encoder without masking to produce full spatial feature maps,
    compatible with Terratorch's PixelWiseModel / ScalarOutputModel pipelines.

    Args:
        encoder: Clay v1.5 Encoder.
        platform: Key in `metadata` that gives the band wavelengths and GSD of the input.
        metadata: Platform metadata, see `CLAY_METADATA`.
        out_indices: Transformer blocks whose outputs are returned. Default [-1], the last block after the
            final LayerNorm. Intermediate blocks are returned without the final LayerNorm.
    """

    def __init__(
        self, encoder: nn.Module, platform: str | list[str], metadata: Box, out_indices: list[int] | None = None
    ):
        super().__init__()
        self.encoder_module = encoder
        self.platform = platform[0] if isinstance(platform, list) else platform
        self.metadata = metadata
        self.patch_size = encoder.patch_size
        self.dim = encoder.dim
        self.out_indices = list(out_indices) if out_indices else [-1]
        self.out_channels = [self.dim] * len(self.out_indices)

    def forward(
        self, x: torch.Tensor, time: torch.Tensor | None = None, latlon: torch.Tensor | None = None, **kwargs
    ) -> list[torch.Tensor]:
        """`time` [B, 4] and `latlon` [B, 4] are Clay's normalized (sin, cos) week/hour and lat/lon; zero if None."""
        B, C, H, W = x.shape
        grid = (H // self.patch_size, W // self.patch_size)

        waves = torch.tensor(
            list(self.metadata[self.platform].bands.wavelength.values()),
            device=x.device,
        )
        gsd = torch.tensor(self.metadata[self.platform].gsd, device=x.device)
        time = torch.zeros(B, 4, device=x.device) if time is None else time
        latlon = torch.zeros(B, 4, device=x.device) if latlon is None else latlon

        # Patch embedding + positional encoding (no masking for downstream tasks)
        patches, _ = self.encoder_module.to_patch_embed(x, waves)
        patches = self.encoder_module.add_encodings(patches, time, latlon, gsd, grid=grid)

        # Prepend CLS token and pass through transformer, keeping every block output
        cls_tokens = repeat(self.encoder_module.cls_token, "1 1 D -> B 1 D", B=B)
        x = torch.cat((cls_tokens, patches), dim=1)
        transformer = self.encoder_module.transformer
        outs = []
        for attn, ff in transformer.layers:
            x = attn(x) + x
            x = ff(x) + x
            outs.append(x)
        outs[-1] = transformer.norm(outs[-1])

        # Strip CLS token and reshape to spatial [B, D, H', W']
        return [rearrange(outs[i][:, 1:], "B (h w) D -> B D h w", h=grid[0], w=grid[1]) for i in self.out_indices]


def _build_backbone(
    platform: str | list[str],
    metadata: dict | None = None,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    checkpoint_path: str | None = None,
    out_indices: list[int] | None = None,
    **encoder_kwargs,
) -> ClayMAEBackbone:
    metadata = Box(metadata or CLAY_METADATA)
    platform_str = platform[0] if isinstance(platform, list) else platform
    if platform_str not in metadata:
        raise KeyError(f"Platform '{platform_str}' not found in metadata. Available: {list(metadata.keys())}")
    encoder = Encoder(**{"mask_ratio": 0.0, "shuffle": False, **CLAY_V1_5_LARGE, **encoder_kwargs})
    if pretrained:
        _load_encoder_weights(encoder, _resolve_checkpoint(checkpoint_path))
    return ClayMAEBackbone(encoder, platform, metadata, out_indices)


@TERRATORCH_BACKBONE_REGISTRY.register
def clay_v1_5_large(
    pretrained: bool = False,  # noqa: FBT001, FBT002
    platform: str = "sentinel-2-l2a",
    metadata: dict | None = None,
    checkpoint_path: str | None = None,
    out_indices: list[int] | None = None,
) -> ClayMAEBackbone:
    """Clay v1.5 ViT-L encoder. `platform` selects the sensor in `metadata` (default: Clay's `CLAY_METADATA`).

    Pretrained weights come from `checkpoint_path` or, if None, from made-with-clay/Clay on Hugging Face.
    """
    return _build_backbone(platform, metadata, pretrained, checkpoint_path, out_indices)


@MODEL_FACTORY_REGISTRY.register
class Clay1_5ModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        bands: list[int] = [],
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> Model:
        """Model factory for Clay v1.5 downstream tasks.

        Builds a full encoder-decoder model using the Clay v1.5 encoder as backbone,
        wrapped in Terratorch's standard PixelWiseModel or ScalarOutputModel pipeline.

        Required kwargs:
            platform (str | list[str]): Platform name matching a key in metadata.

        Optional kwargs:
            metadata (dict): Platform metadata containing band wavelengths and GSD. Default: Clay's
                `CLAY_METADATA` (sentinel-2-l2a, sentinel-1-rtc, landsat-c2l1, landsat-c2l2-sr, naip, linz,
                planetscope-sr, modis, satellogic-MSI-L1D).
            dim, depth, heads, dim_head, mlp_ratio, patch_size: Encoder size. Default: the released
                ViT-L (`CLAY_V1_5_LARGE`).
            out_indices (list[int]): Transformer blocks whose outputs feed the decoder. Default [-1].
            mask_ratio (float): Unused for inference; kept for compatibility. Default 0.75.
            shuffle (bool): Unused for inference; kept for compatibility. Default False.
            padding (str): Padding mode for PixelWiseModel. Default "reflect".
            batch_size (int): Ignored; accepted for backward compatibility.
            decoder_* (kwargs): Passed to the terratorch decoder constructor.
                Note: Clay v1.5 internal keys (decoder_dim, decoder_depth, decoder_heads,
                decoder_dim_head, decoder_mlp_ratio) are silently ignored.

        Args:
            task: One of "segmentation", "regression", "classification", "scalar_regression".
            backbone: Ignored; kept for API compatibility with other factories.
            decoder: Decoder class name (e.g. "FCNDecoder") or an nn.Module instance.
            in_channels: Number of input channels.
            bands: Band names; informational only.
            num_classes: Number of output classes. Required for segmentation/classification.
            pretrained: If True, loads weights from checkpoint_path or downloads from HuggingFace.
            num_frames: Ignored; kept for API compatibility.
            aux_decoders: Optional list of auxiliary decoder heads.
            rescale: Rescale model output to input spatial size via bilinear interpolation.
            checkpoint_path: Path to a local Clay v1.5 checkpoint (.ckpt or .pt).
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)

        platform = kwargs.get("platform")
        if platform is None:
            raise ValueError("'platform' is required in kwargs for Clay1_5ModelFactory")

        patch_size = kwargs.get("patch_size", CLAY_V1_5_LARGE["patch_size"])
        padding = kwargs.get("padding", "reflect")

        encoder_keys = [*CLAY_V1_5_LARGE, "mask_ratio", "shuffle"]
        backbone_module = _build_backbone(
            platform,
            kwargs.get("metadata"),
            pretrained,
            checkpoint_path,
            kwargs.get("out_indices"),
            **{k: kwargs[k] for k in encoder_keys if k in kwargs},
        )
        feature_channels = backbone_module.out_channels

        decoder_cls = _get_decoder(decoder)
        decoder_kwargs, _ = extract_prefix_keys(kwargs, "decoder_")
        decoder_instance = decoder_cls(feature_channels, **_strip_clay_keys(decoder_kwargs))

        head_kwargs, _ = extract_prefix_keys(kwargs, "head_")
        if num_classes:
            head_kwargs["num_classes"] = num_classes

        if aux_decoders is None:
            return _build_appropriate_model(
                task, backbone_module, decoder_instance, head_kwargs,
                patch_size=patch_size, padding=padding, rescale=rescale,
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls = _get_decoder(aux_decoder.decoder)
            aux_decoder_kwargs, _ = extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(feature_channels, **_strip_clay_keys(aux_decoder_kwargs))
            aux_head_kwargs, _ = extract_prefix_keys(args, "head_")
            if num_classes:
                aux_head_kwargs["num_classes"] = num_classes
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(
                    aux_decoder.name, aux_decoder_instance, aux_head_kwargs
                )
            )

        return _build_appropriate_model(
            task, backbone_module, decoder_instance, head_kwargs,
            patch_size=patch_size, padding=padding, rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _resolve_checkpoint(checkpoint_path: str | None) -> str:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(path)
    # No explicit path provided — try HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading Clay v1.5 checkpoint from HuggingFace...")
        return hf_hub_download(repo_id="made-with-clay/Clay", filename="v1.5/clay-v1.5.ckpt")
    except Exception as e:
        msg = "Could not download the Clay v1.5 checkpoint; pass checkpoint_path or pretrained=False"
        raise RuntimeError(msg) from e


def _load_encoder_weights(encoder: nn.Module, ckpt_path: str) -> None:
    logger.info(f"Loading encoder weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Clay v1.5 checkpoints store the full ClayMAE under "model.encoder.*"
    prefix = "model.encoder."
    encoder_state = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    if not encoder_state:
        msg = f"No keys with prefix '{prefix}' found in checkpoint. Keys found: {list(state_dict.keys())[:10]} ..."
        raise KeyError(msg)
    # strict: a wrong encoder size or checkpoint layout must fail, not leave weights random
    encoder.load_state_dict(encoder_state)
    logger.info("Encoder weights loaded successfully")


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    patch_size: int | None,
    padding: str,
    rescale: bool = True,
    auxiliary_heads=None,
):
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task, backbone, decoder, head_kwargs,
            patch_size=patch_size, padding=padding, rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    if task in SCALAR_TASKS:
        # rescale is not passed: ScalarOutputModel does not do spatial output
        return ScalarOutputModel(
            task, backbone, decoder, head_kwargs,
            patch_size=patch_size, padding=padding,
            auxiliary_heads=auxiliary_heads,
        )
    raise AssertionError(f"Unreachable: unexpected task '{task}'")


def _get_decoder(decoder: str | nn.Module) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    if isinstance(decoder, str):
        try:
            return getattr(decoder_registry, decoder)
        except AttributeError as e:
            msg = f"Decoder {decoder} was not found in the registry."
            raise DecoderNotFoundError(msg) from e
    msg = "Decoder must be str or nn.Module"
    raise Exception(msg)
