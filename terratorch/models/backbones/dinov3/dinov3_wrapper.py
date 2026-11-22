import torch
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

# Input normalization of the DINOv3 weights (RGB in [0, 1]): LVD-1689M uses ImageNet stats, SAT-493M its own.
DINOV3_LVD_MEAN = [0.485, 0.456, 0.406]
DINOV3_LVD_STD = [0.229, 0.224, 0.225]
DINOV3_SAT_MEAN = [0.430, 0.411, 0.296]
DINOV3_SAT_STD = [0.213, 0.156, 0.143]


class DinoV3Wrapper(nn.Module):
    def __init__(
        self,
        model: str,
        ckpt_path: str = None,
        return_cls_token: bool = True,
        out_indices: list[int] | None = None,
        reshape: bool = False,
        repo_or_dir: str = "facebookresearch/dinov3",
        **kwargs,
    ):
        """
        TerraTorch wrapper for DINO V3 models.

        Args:
        model: model name from the ones supported in https://github.com/facebookresearch/dinov3/blob/main/hubconf.py
        ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
        return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.
        out_indices: indices of the ViT blocks or ConvNeXt stages to return. Defaults to all of them.
        reshape: return [B, C, H, W] feature maps instead of token sequences (no class token).
            ConvNeXt stages have strides 4, 8, 16 and 32.
        repo_or_dir: `torch.hub.load` repo, optionally pinned (`facebookresearch/dinov3:<ref>`),
            or a local clone together with `source="local"`.
        **kwargs: forwarded to `torch.hub.load` and the DINOv3 hub entrypoint, e.g. `source`, or `patch_size` for ConvNeXt
            (resizes all stage maps to the grid of a ViT with that patch size).

        Returns:
            DinoV3Wrapper model

        """

        super().__init__()
        # Weights come from ckpt_path only. The hub ViT entrypoints fix img_size; RoPE handles any input size.
        kwargs.pop("pretrained", None)
        kwargs.pop("img_size", None)
        if ckpt_path is None:
            self.dinov3 = torch.hub.load(repo_or_dir, model, pretrained=False, **kwargs)
        else:
            self.dinov3 = torch.hub.load(repo_or_dir, model, weights=ckpt_path, **kwargs)

        if hasattr(self.dinov3, "blocks"):
            dims = [self.dinov3.embed_dim] * len(self.dinov3.blocks)
        else:
            dims = self.dinov3.embed_dims
        self.out_indices = list(range(len(dims))) if out_indices is None else list(out_indices)
        self.out_channels = [dims[i] for i in self.out_indices]
        self.return_cls_token = return_cls_token
        self.reshape = reshape

    def forward(self, x: Tensor):
        """
        Forward pass for the model.

        Args:
            x: tensor of shape Bx3xHxW

        Returns:
            list of embeddings for the blocks in out_indices: [B, N, C] tokens, or [B, C, H, W] maps if reshape.
        """

        return_cls_token = self.return_cls_token and not self.reshape
        feats = self.dinov3.get_intermediate_layers(
            x, n=self.out_indices, reshape=self.reshape, return_class_token=return_cls_token
        )

        if return_cls_token:
            # rearrange with class token in front
            feats = [torch.cat([x[1].unsqueeze(1), x[0]], axis=1) for x in feats]

        return list(feats)


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vits16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vits16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vits16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vits16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vits16plus(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vits16plus model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vits16plus model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vits16plus", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitb16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vitb16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vitb16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vitb16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitl16plus(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vitl16plus model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vitl16plus model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vitl16plus", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitl16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vitl16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vitl16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vitl16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vith16plus(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vith16plus model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vith16plus model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vith16plus", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vit7b16(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_vit7b16 model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_vit7b16 model

    """

    # Init model
    model = DinoV3Wrapper(model="dinov3_vit7b16", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs)

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_base(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_base model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_base model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_base", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_large(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_large model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_large model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_large", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_small(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_small model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_small model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_small", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_convnext_tiny(ckpt_path: str, return_cls_token: bool = True, **kwargs):
    """
    Constructor for the dinov3_convnext_tiny model

    Args:
    ckpt_path: local path to the desired dino v3 model downloaded from Meta (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
    return_cls_token: whether the model should return the class token or not. If True, the class token will be the first token in the embedding sequence.

    Returns:
        DinoV3Wrapper dinov3_convnext_tiny model

    """

    # Init model
    model = DinoV3Wrapper(
        model="dinov3_convnext_tiny", ckpt_path=ckpt_path, return_cls_token=return_cls_token, **kwargs
    )

    return model
