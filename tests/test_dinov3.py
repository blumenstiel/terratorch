import gc
from pathlib import Path

import pytest
import torch

from terratorch.models.backbones.dinov3.dinov3_wrapper import DinoV3Wrapper


# Fixture for dummy input
@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 224, 224)  # Batch size 1, RGB, 224x224


# Fixture for model
@pytest.fixture
def model():
    return DinoV3Wrapper(model="dinov3_vits16", img_size=224)


# Test model initialization
def test_model_initialization(model):
    assert model.dinov3 is not None
    assert hasattr(model.dinov3, "get_intermediate_layers")


# Test forward pass
def test_forward_features(model, dummy_input):
    output = model.forward(dummy_input)
    assert isinstance(output, list)
    assert isinstance(output[0], torch.Tensor)
    # batch size match
    gc.collect()


# Test output shape consistency
def test_output_shape(model, dummy_input):
    output = model.forward(dummy_input)
    # Depending on the model, output shape may vary. Adjust as needed.
    assert len(output) == 12
    assert output[0].shape[0] == 1
    assert output[0].shape[1] == 197
    assert output[0].shape[2] == 384

    model.return_cls_token = False

    output = model.forward(dummy_input)
    assert output[0].shape[1] == 196
    gc.collect()


# Test model with different image size
def test_model_different_img_size():
    model = DinoV3Wrapper(model="dinov3_vits16", img_size=256)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model.forward(dummy_input)
    assert output[0].shape[0] == 1
    gc.collect()


# Download-free tests against a local clone of the DINOv3 hub repo (random weights)
HUB_DIR = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
local_hub = pytest.mark.skipif(not (HUB_DIR / "hubconf.py").exists(), reason="no local DINOv3 hub repo")


def _local(model, **kwargs):
    return DinoV3Wrapper(model=model, repo_or_dir=str(HUB_DIR), source="local", **kwargs)


@local_hub
def test_cls_token_first():
    model = _local("dinov3_vits16")
    x = torch.randn(1, 3, 64, 64)
    output = model(x)
    patches, cls = model.dinov3.get_intermediate_layers(x, n=[11], return_class_token=True)[0]
    torch.testing.assert_close(output[-1][:, 0], cls, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(output[-1][:, 1:], patches, atol=1e-4, rtol=1e-3)


@local_hub
def test_out_indices_and_reshape():
    model = _local("dinov3_vits16", out_indices=[3, 11], reshape=True)
    output = model(torch.randn(1, 3, 64, 64))
    assert model.out_channels == [384, 384]
    assert [o.shape for o in output] == [torch.Size([1, 384, 4, 4])] * 2


@local_hub
def test_convnext_stride16_stage_map():
    model = _local("dinov3_convnext_tiny", out_indices=[2], reshape=True)
    output = model(torch.randn(1, 3, 64, 64))
    assert model.out_channels == [384]
    assert len(output) == 1
    assert output[0].shape == (1, 384, 4, 4)


@local_hub
def test_kwargs_forwarded_to_hub():
    # ConvNeXt patch_size resizes every stage to the H/16 grid; factories also pass pretrained and img_size
    model = _local("dinov3_convnext_tiny", return_cls_token=False, patch_size=16, pretrained=True, img_size=64)
    output = model(torch.randn(1, 3, 64, 64))
    assert [o.shape[1] for o in output] == [16] * 4
