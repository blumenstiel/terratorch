import json
import gc
import matplotlib.pyplot as plt
import pytest
from utils import create_dummy_h5_pickle

from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_mpv4ger_data(tmp_path) -> str:
    base_dir = tmp_path / "mpv4ger"
    base_dir.mkdir()
    data_dir = base_dir / "m-pv4ger"
    data_dir.mkdir()
    partition_file = data_dir / "default_partition.json"
    dummy_partitions = {
        "train": ["sample_train"],
        "valid": ["sample_valid"],
        "test": ["sample_test"]
    }
    with open(partition_file, "w") as f:
        json.dump(dummy_partitions, f)
    band_keys = list(("BLUE", "GREEN", "RED"))
    dummy_shape = (256, 256)
    for sample in ["sample_train", "sample_valid", "sample_test"]:
        file_path = data_dir / f"{sample}.hdf5"
        create_dummy_h5_pickle(str(file_path), band_keys, dummy_shape, label=1)
    return str(base_dir)

def test_mpv4ger_datamodule(dummy_mpv4ger_data):
    from terratorch.datamodules import MPv4gerNonGeoDataModule
    from terratorch.datasets import MPv4gerNonGeo

    bands = MPv4gerNonGeo.all_band_names
    datamodule = MPv4gerNonGeoDataModule(
        data_root=dummy_mpv4ger_data,
        bands=bands,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "label" in train_batch, "Missing key 'label' in train batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in validation batch"
    assert "label" in val_batch, "Missing key 'label' in validation batch"
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "label" in test_batch, "Missing key 'label' in test batch"
    datamodule.setup("predict")
    predict_loader = datamodule.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "image" in predict_batch, "Missing key 'image' in predict batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    sample = unbind_samples(val_batch)[0]
    datamodule.plot(sample)
    plt.close()
    gc.collect()
