"""Tests for Clay v1.5 model factory and backbone adapter.

Covers ClayMAEBackbone (the encoder wrapper) and Clay1_5ModelFactory
for segmentation, regression, and classification tasks.
"""

import pytest
import torch
from box import Box

from terratorch.models.clay1_5_model_factory import Clay1_5ModelFactory, ClayMAEBackbone
from terratorch.models.backbones.clay_v15.model import Encoder
from terratorch.models.model import ModelOutput
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NAIP_METADATA = {
    "naip": {
        "band_order": ["red", "green", "blue", "nir"],
        "rgb_indices": [0, 1, 2],
        "gsd": 1.0,
        "bands": {
            "mean": {"red": 110.16, "green": 115.41, "blue": 98.15, "nir": 139.04},
            "std": {"red": 47.23, "green": 39.82, "blue": 35.43, "nir": 49.86},
            "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir": 0.842},
        },
    }
}

# Tiny encoder config so tests run fast without a GPU
TINY_ENCODER_KWARGS = dict(
    dim=64,
    depth=2,
    heads=2,
    dim_head=32,
    mlp_ratio=2,
    patch_size=8,
)

# Minimal kwargs for the factory (no pretrained weights, no HuggingFace download)
BASE_FACTORY_KWARGS = dict(
    **TINY_ENCODER_KWARGS,
    metadata=NAIP_METADATA,
    platform=["naip"],
    pretrained=False,
)


def _make_encoder() -> Encoder:
    return Encoder(mask_ratio=0.75, shuffle=False, **TINY_ENCODER_KWARGS)


def _make_backbone() -> ClayMAEBackbone:
    return ClayMAEBackbone(
        encoder=_make_encoder(),
        platform=["naip"],
        metadata=Box(NAIP_METADATA),
    )


# ---------------------------------------------------------------------------
# ClayMAEBackbone tests
# ---------------------------------------------------------------------------

class TestClayMAEBackbone:
    """Tests for the ClayMAEBackbone encoder adapter."""

    def test_initialization(self):
        backbone = _make_backbone()

        assert backbone.platform == "naip"
        assert backbone.dim == TINY_ENCODER_KWARGS["dim"]
        assert backbone.patch_size == TINY_ENCODER_KWARGS["patch_size"]

    def test_platform_string_accepted(self):
        backbone = ClayMAEBackbone(
            encoder=_make_encoder(),
            platform="naip",
            metadata=Box(NAIP_METADATA),
        )
        assert backbone.platform == "naip"

    def test_forward_returns_list(self):
        backbone = _make_backbone()
        x = torch.randn(1, 4, 64, 64)
        out = backbone(x)

        assert isinstance(out, list)
        assert len(out) == 1

    def test_forward_output_shape(self):
        backbone = _make_backbone()
        B, C, H, W = 1, 4, 64, 64
        x = torch.randn(B, C, H, W)
        out = backbone(x)

        expected_grid = H // TINY_ENCODER_KWARGS["patch_size"]
        assert out[0].shape == (B, TINY_ENCODER_KWARGS["dim"], expected_grid, expected_grid)

    def test_forward_batch_size_greater_than_one(self):
        """Core regression test: the original fix had a bug where only batch_size=1 worked
        because encoded_patches[-1] indexed the last batch item instead of removing the
        CLS token. This test ensures the fix works for any batch size."""
        backbone = _make_backbone()
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 4, 64, 64)
            out = backbone(x)
            expected_grid = 64 // TINY_ENCODER_KWARGS["patch_size"]
            assert out[0].shape == (batch_size, TINY_ENCODER_KWARGS["dim"], expected_grid, expected_grid), \
                f"Wrong shape for batch_size={batch_size}"

    def test_forward_output_is_finite(self):
        backbone = _make_backbone()
        x = torch.randn(2, 4, 64, 64)
        out = backbone(x)
        assert torch.isfinite(out[0]).all()

    def test_forward_feature_count_matches_all_patches(self):
        """Spatial grid must equal full unmasked patch count (not a masked subset)."""
        backbone = _make_backbone()
        H, W = 64, 64
        patch_size = TINY_ENCODER_KWARGS["patch_size"]
        x = torch.randn(1, 4, H, W)
        with torch.no_grad():
            out = backbone(x)
        features = out[0]
        assert features.shape[2] * features.shape[3] == (H // patch_size) * (W // patch_size)

    def test_gradient_flow(self):
        backbone = _make_backbone()
        x = torch.randn(2, 4, 64, 64, requires_grad=True)
        out = backbone(x)
        loss = out[0].sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_eval_mode_deterministic(self):
        backbone = _make_backbone()
        backbone.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out1 = backbone(x)
            out2 = backbone(x)
        assert torch.allclose(out1[0], out2[0])

    def test_batch_independence(self):
        """Each sample in a batch should be processed independently."""
        backbone = _make_backbone()
        backbone.eval()
        x1 = torch.randn(1, 4, 64, 64)
        x2 = torch.randn(1, 4, 64, 64)
        x_batched = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out1 = backbone(x1)
            out2 = backbone(x2)
            out_batched = backbone(x_batched)

        assert torch.allclose(out_batched[0][0], out1[0][0], atol=1e-5)
        assert torch.allclose(out_batched[0][1], out2[0][0], atol=1e-5)

    def test_different_spatial_sizes(self):
        backbone = _make_backbone()
        for h, w in [(64, 64), (128, 128)]:
            x = torch.randn(1, 4, h, w)
            out = backbone(x)
            expected_grid_h = h // TINY_ENCODER_KWARGS["patch_size"]
            expected_grid_w = w // TINY_ENCODER_KWARGS["patch_size"]
            assert out[0].shape == (1, TINY_ENCODER_KWARGS["dim"], expected_grid_h, expected_grid_w)


# ---------------------------------------------------------------------------
# Clay1_5ModelFactory tests
# ---------------------------------------------------------------------------

class TestClay1_5ModelFactory:
    """Tests for Clay1_5ModelFactory.build_model()."""

    @pytest.fixture
    def factory(self):
        return Clay1_5ModelFactory()

    def test_segmentation_returns_pixel_wise_model(self, factory):
        model = factory.build_model(
            task="segmentation",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=10,
            **BASE_FACTORY_KWARGS,
        )
        assert isinstance(model, PixelWiseModel)

    def test_regression_returns_pixel_wise_model(self, factory):
        model = factory.build_model(
            task="regression",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            **BASE_FACTORY_KWARGS,
        )
        assert isinstance(model, PixelWiseModel)

    def test_classification_returns_scalar_output_model(self, factory):
        model = factory.build_model(
            task="classification",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=5,
            **BASE_FACTORY_KWARGS,
        )
        assert isinstance(model, ScalarOutputModel)

    def test_unsupported_task_raises(self, factory):
        with pytest.raises(NotImplementedError):
            factory.build_model(
                task="invalid_task",
                backbone="clay15",
                decoder="FCNDecoder",
                in_channels=4,
                **BASE_FACTORY_KWARGS,
            )

    def test_task_is_case_insensitive(self, factory):
        model = factory.build_model(
            task="SEGMENTATION",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=10,
            **BASE_FACTORY_KWARGS,
        )
        assert isinstance(model, PixelWiseModel)

    def test_invalid_decoder_raises(self, factory):
        with pytest.raises(Exception):
            factory.build_model(
                task="segmentation",
                backbone="clay15",
                decoder="NonExistentDecoder",
                in_channels=4,
                num_classes=10,
                **BASE_FACTORY_KWARGS,
            )

    def test_backbone_is_clay_mae_backbone(self, factory):
        model = factory.build_model(
            task="segmentation",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=10,
            **BASE_FACTORY_KWARGS,
        )
        assert isinstance(model.encoder, ClayMAEBackbone)

    def test_scalar_regression_returns_scalar_output_model(self, factory):
        model = factory.build_model(
            task="scalar_regression",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=1,
            **BASE_FACTORY_KWARGS,
        )
        assert isinstance(model, ScalarOutputModel)

    def test_missing_platform_raises(self, factory):
        kwargs = {k: v for k, v in BASE_FACTORY_KWARGS.items() if k != "platform"}
        with pytest.raises(ValueError, match="platform"):
            factory.build_model(
                task="segmentation",
                backbone="clay15",
                decoder="FCNDecoder",
                in_channels=4,
                num_classes=10,
                **kwargs,
            )

    def test_missing_metadata_uses_clay_metadata(self, factory):
        kwargs = {k: v for k, v in BASE_FACTORY_KWARGS.items() if k != "metadata"}
        model = factory.build_model(
            task="segmentation",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=10,
            **kwargs,
        )
        assert model.encoder.metadata.naip.gsd == 1.0
        assert model(torch.randn(1, 4, 64, 64)).output.shape == (1, 10, 64, 64)

    def test_platform_not_in_metadata_raises(self, factory):
        with pytest.raises(KeyError):
            factory.build_model(
                task="segmentation",
                backbone="clay15",
                decoder="FCNDecoder",
                in_channels=4,
                num_classes=10,
                **{**BASE_FACTORY_KWARGS, "platform": "nonexistent"},
            )

    def test_clay_internal_decoder_kwargs_ignored(self, factory):
        """Old ClayMAE decoder kwargs (decoder_dim, decoder_depth, etc.) must not
        cause errors — they should be silently dropped."""
        model = factory.build_model(
            task="segmentation",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=10,
            # These are Clay v1.5 internal args that should be ignored
            decoder_dim=96,
            decoder_depth=3,
            decoder_heads=2,
            decoder_dim_head=48,
            decoder_mlp_ratio=2,
            **BASE_FACTORY_KWARGS,
        )
        assert isinstance(model, PixelWiseModel)


# ---------------------------------------------------------------------------
# End-to-end forward pass tests
# ---------------------------------------------------------------------------

class TestClay1_5ModelFactoryForward:
    """End-to-end forward pass tests for models built by Clay1_5ModelFactory."""

    @pytest.fixture
    def segmentation_model(self):
        return Clay1_5ModelFactory().build_model(
            task="segmentation",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=10,
            **BASE_FACTORY_KWARGS,
        )

    def test_forward_returns_model_output(self, segmentation_model):
        """The model must return a ModelOutput, not a tuple of losses."""
        x = torch.randn(1, 4, 64, 64)
        out = segmentation_model(x)
        assert isinstance(out, ModelOutput)

    def test_forward_output_has_correct_num_classes(self, segmentation_model):
        x = torch.randn(1, 4, 64, 64)
        out = segmentation_model(x)
        assert out.output.shape[1] == 10

    def test_forward_output_spatial_size_matches_input(self, segmentation_model):
        """With rescale=True, output spatial dims should match input."""
        H, W = 64, 64
        x = torch.randn(1, 4, H, W)
        out = segmentation_model(x)
        assert out.output.shape[-2:] == (H, W)

    def test_forward_batch_size_one(self, segmentation_model):
        x = torch.randn(1, 4, 64, 64)
        out = segmentation_model(x)
        assert out.output.shape[0] == 1

    def test_forward_batch_size_two(self, segmentation_model):
        """Regression test for the batch_size > 1 bug in the original implementation."""
        x = torch.randn(2, 4, 64, 64)
        out = segmentation_model(x)
        assert out.output.shape[0] == 2

    def test_forward_output_is_finite(self, segmentation_model):
        x = torch.randn(2, 4, 64, 64)
        out = segmentation_model(x)
        assert torch.isfinite(out.output).all()

    def test_forward_gradient_flow(self, segmentation_model):
        x = torch.randn(2, 4, 64, 64, requires_grad=True)
        out = segmentation_model(x)
        loss = out.output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_forward_regression_task(self):
        model = Clay1_5ModelFactory().build_model(
            task="regression",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            **BASE_FACTORY_KWARGS,
        )
        x = torch.randn(2, 4, 64, 64)
        out = model(x)
        assert isinstance(out, ModelOutput)
        assert torch.isfinite(out.output).all()

    def test_forward_different_spatial_sizes(self, segmentation_model):
        for h, w in [(64, 64), (128, 128)]:
            x = torch.randn(1, 4, h, w)
            out = segmentation_model(x)
            assert out.output.shape[-2:] == (h, w), f"Failed for input size ({h},{w})"

    def test_forward_eval_mode_deterministic(self, segmentation_model):
        segmentation_model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            out1 = segmentation_model(x)
            out2 = segmentation_model(x)
        assert torch.allclose(out1.output, out2.output)

    def test_backward_compatible_with_extra_clay_kwargs(self):
        """Old configs passing ClayMAE-specific kwargs should still work."""
        model = Clay1_5ModelFactory().build_model(
            task="segmentation",
            backbone="clay15",
            decoder="FCNDecoder",
            in_channels=4,
            num_classes=5,
            # Legacy ClayMAE kwargs
            mask_ratio=0.75,
            norm_pix_loss=False,
            shuffle=True,
            teacher="vit_large_patch14_reg4_dinov2.lvd142m",
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            batch_size=1,
            decoder_dim=96,
            decoder_depth=3,
            decoder_heads=2,
            decoder_dim_head=48,
            decoder_mlp_ratio=2,
            **BASE_FACTORY_KWARGS,
        )
        x = torch.randn(1, 4, 64, 64)
        out = model(x)
        assert isinstance(out, ModelOutput)
        assert out.output.shape == (1, 5, 64, 64)

    def test_model_is_in_training_mode_by_default(self, segmentation_model):
        """Model should be in training mode after construction, ready for fine-tuning."""
        assert segmentation_model.training is True


# ---------------------------------------------------------------------------
# Checkpoint loading tests
# ---------------------------------------------------------------------------

class TestCheckpointLoading:
    """Tests for _resolve_checkpoint and _load_encoder_weights."""

    def test_resolve_checkpoint_valid_path(self, tmp_path):
        from terratorch.models.clay1_5_model_factory import _resolve_checkpoint
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"")
        result = _resolve_checkpoint(str(ckpt))
        assert result == str(ckpt)

    def test_resolve_checkpoint_invalid_path_raises(self):
        from terratorch.models.clay1_5_model_factory import _resolve_checkpoint
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            _resolve_checkpoint("/nonexistent/path/model.ckpt")

    def test_load_encoder_weights_wrong_prefix_raises(self, tmp_path):
        from terratorch.models.clay1_5_model_factory import _load_encoder_weights
        # Checkpoint with wrong key prefix — no "model.encoder." keys
        fake_state = {"encoder.some_weight": torch.zeros(4)}
        ckpt_path = tmp_path / "wrong_prefix.ckpt"
        torch.save({"state_dict": fake_state}, str(ckpt_path))
        with pytest.raises(KeyError, match="model.encoder."):
            _load_encoder_weights(_make_encoder(), str(ckpt_path))

    def test_load_encoder_weights_wrong_depth_raises(self, tmp_path):
        from terratorch.models.clay1_5_model_factory import _load_encoder_weights
        deeper = Encoder(mask_ratio=0.75, shuffle=False, **{**TINY_ENCODER_KWARGS, "depth": 3})
        ckpt_path = tmp_path / "deeper.ckpt"
        torch.save({"state_dict": {f"model.encoder.{k}": v for k, v in deeper.state_dict().items()}}, str(ckpt_path))
        with pytest.raises(RuntimeError, match="Unexpected key"):
            _load_encoder_weights(_make_encoder(), str(ckpt_path))

    def test_resolve_checkpoint_download_failure_raises(self, monkeypatch):
        import huggingface_hub
        from terratorch.models.clay1_5_model_factory import _resolve_checkpoint

        def offline(*args, **kwargs):
            raise OSError("offline")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
        with pytest.raises(RuntimeError, match="checkpoint"):
            _resolve_checkpoint(None)

    def test_load_encoder_weights_correct_prefix(self, tmp_path, caplog):
        import logging
        from terratorch.models.clay1_5_model_factory import _load_encoder_weights
        encoder = _make_encoder()
        # Real Clay v1.5 checkpoint layout: "model.encoder.<key>"
        encoder_state = encoder.state_dict()
        full_state = {f"model.encoder.{k}": v for k, v in encoder_state.items()}
        ckpt_path = tmp_path / "correct.ckpt"
        torch.save({"state_dict": full_state}, str(ckpt_path))
        with caplog.at_level(logging.INFO, logger="terratorch"):
            _load_encoder_weights(encoder, str(ckpt_path))
        assert "NOT loaded" not in caplog.text
        assert "loaded successfully" in caplog.text


# ---------------------------------------------------------------------------
# Registered backbone, metadata, time / latlon, non-square input, out_indices
# ---------------------------------------------------------------------------

class TestClayV15Backbone:
    def test_registered_large_backbone(self):
        from terratorch.registry import BACKBONE_REGISTRY
        backbone = BACKBONE_REGISTRY.build("clay_v1_5_large", pretrained=False)
        assert backbone.platform == "sentinel-2-l2a"
        assert backbone.out_channels == [1024]
        assert len(backbone.encoder_module.transformer.layers) == 24
        assert backbone.patch_size == 8

    def test_clay_metadata(self):
        from terratorch.models.backbones.clay_v15.metadata import CLAY_METADATA
        s2 = CLAY_METADATA["sentinel-2-l2a"]
        assert s2["gsd"] == 10.0
        assert list(s2["bands"]["wavelength"]) == s2["band_order"]
        assert CLAY_METADATA["sentinel-1-rtc"]["bands"]["wavelength"] == {"vv": 3.5, "vh": 4.0}

    def test_time_latlon_kwargs(self):
        backbone = _make_backbone().eval()
        x = torch.randn(2, 4, 32, 32)
        zeros = torch.zeros(2, 4)
        latlon = torch.tensor([[0.5, 0.8, -0.2, 0.9]] * 2)
        with torch.no_grad():
            default = backbone(x)[0]
            assert torch.equal(default, backbone(x, time=zeros, latlon=zeros)[0])
            assert not torch.allclose(default, backbone(x, time=zeros, latlon=latlon)[0])

    def test_non_square_input(self):
        backbone = _make_backbone().eval()
        out = backbone(torch.randn(1, 4, 32, 64))
        assert out[0].shape == (1, TINY_ENCODER_KWARGS["dim"], 4, 8)

    def test_non_square_matches_square_crop(self):
        # Row-major patch order: the top-left 4x4 patches of a 4x8 grid get the same position encoding as a 4x4 grid
        encoder = _make_encoder()
        patches = torch.zeros(1, 32, TINY_ENCODER_KWARGS["dim"])
        gsd = torch.tensor(1.0)
        wide = encoder.add_encodings(patches, torch.zeros(1, 4), torch.zeros(1, 4), gsd, grid=(4, 8))
        square = encoder.add_encodings(patches[:, :16], torch.zeros(1, 4), torch.zeros(1, 4), gsd)
        assert torch.equal(wide.reshape(1, 4, 8, -1)[:, :, :4].reshape(1, 16, -1), square)

    def test_out_indices(self):
        encoder = _make_encoder()
        default = ClayMAEBackbone(encoder, "naip", Box(NAIP_METADATA)).eval()
        multi = ClayMAEBackbone(encoder, "naip", Box(NAIP_METADATA), out_indices=[0, -1]).eval()
        assert multi.out_channels == [TINY_ENCODER_KWARGS["dim"]] * 2
        x = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            first, last = multi(x)
            assert torch.equal(last, default(x)[0])
            assert not torch.allclose(first, last)

    def test_factory_out_indices_feed_decoder(self):
        model = Clay1_5ModelFactory().build_model(
            task="segmentation",
            backbone="clay15",
            decoder="UNetDecoder",
            decoder_channels=[32, 16],
            in_channels=4,
            num_classes=3,
            out_indices=[0, 1],
            **BASE_FACTORY_KWARGS,
        )
        assert model(torch.randn(1, 4, 32, 32)).output.shape == (1, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
