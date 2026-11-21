import pytest
import torch

from terratorch.models.backbones import dofa_vit


class _FakeWeights:
    """Stands in for a torchgeo WeightsEnum member, so no download is needed."""

    def __init__(self, state_dict):
        self.state_dict = state_dict

    def get_state_dict(self, **kwargs):
        return self.state_dict


def _pretrained_224_state_dict():
    model = dofa_vit.dofa.dofa_base_patch16_224()
    sd = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith(("fc_norm", "head"))}
    sd["pos_embed"] = torch.randn_like(sd["pos_embed"])  # pretrained pos_embed of the 224 px model (14 x 14 + CLS)
    return sd


@pytest.mark.parametrize("img_size", [224, 256, 512])
def test_dofa_weights_keep_pos_embed(img_size):
    checkpoint = _pretrained_224_state_dict()
    model = dofa_vit.dofa.dofa_base_patch16_224(img_size=img_size)
    dofa_vit.load_dofa_weights(model, "bilinear", weights=_FakeWeights(checkpoint), input_size=img_size)

    assert model.pos_embed.shape[1] == (img_size // 16) ** 2 + 1
    assert model.pos_embed.abs().sum() > 0  # not left at torchgeo's zero init
    assert torch.equal(model.pos_embed[:, 0], checkpoint["pos_embed"][:, 0])  # CLS position kept
    if img_size == 224:
        assert torch.equal(model.pos_embed, checkpoint["pos_embed"])


def test_dofa_sar_wavelength():
    assert dofa_vit.get_wavelenghts(["VV", "VH"]) == [3.75, 3.75]
