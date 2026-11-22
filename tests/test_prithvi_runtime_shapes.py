import torch

from terratorch.models.backbones.prithvi_mae import PrithviMAE, PrithviViT

TINY = {"patch_size": (1, 8, 8), "in_chans": 3, "embed_dim": 32, "depth": 2, "num_heads": 4}


def _temporal(b, t):
    return torch.tensor([[2020.0, d] for d in range(t)]).expand(b, t, 2)


def test_temporal_coords_runtime_num_frames():
    """temporal_coords work when the input T differs from the constructed num_frames."""
    model = PrithviViT(img_size=16, num_frames=1, coords_encoding=["time"], **TINY)
    x = torch.randn(2, 3, 3, 16, 16)
    feats = model.forward_features(x, temporal_coords=_temporal(2, 3))
    assert feats[-1].shape == (2, 1 + 3 * 2 * 2, 32)
    out, _, _ = model(x, temporal_coords=_temporal(2, 3), mask_ratio=0.5)
    assert out.shape == (2, 1 + 6, 32)


def test_mae_temporal_coords_runtime_num_frames():
    model = PrithviMAE(
        img_size=16, num_frames=4, coords_encoding=["time"], decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2,
        **TINY,
    )
    x = torch.randn(2, 3, 2, 16, 16)
    loss, pred, _ = model(x, temporal_coords=_temporal(2, 2))
    assert pred.shape == x.shape
    assert torch.isfinite(loss["loss"])


def test_prepare_features_non_square():
    model = PrithviViT(img_size=(16, 32), num_frames=2, **TINY)
    x = torch.randn(2, 3, 2, 16, 40)  # runtime size differs from img_size as well
    out = model.prepare_features_for_image_model(model.forward_features(x))
    assert out[-1].shape == (2, 2 * 32, 2, 5)
    # Channel layout is (t e): frame 1 of token row 0, col 1.
    tokens = model.forward_features(x)[-1][:, 1:]
    assert torch.equal(out[-1][:, 32:, 0, 1], tokens[:, 2 * 5 + 1])
