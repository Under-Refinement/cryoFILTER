import torch

from models.predict import _resolve_checkpoint_model_config


def test_frequency_band_checkpoint_metadata_is_preserved() -> None:
    checkpoint = {
        "model_type": "unet_attention",
        "attention_type": "global",
        "use_power_spectrum": True,
        "include_real_space_input": True,
        "input_channels": 5,
        "psd_multiscale": False,
        "psd_multiscale_separate_channels": False,
        "psd_frequency_band_channels": True,
        "psd_frequency_bands": [[0.03, 0.05], [0.09, 0.11], [0.14, 0.16], [0.22, 0.24]],
        "model_state_dict": {
            "module.enc1.0.weight": torch.zeros((64, 5, 3, 3), dtype=torch.float32),
            "module.final.weight": torch.zeros((2, 64, 1, 1), dtype=torch.float32),
        },
    }

    config = _resolve_checkpoint_model_config(checkpoint)

    assert config["model_type"] == "unet_attention"
    assert config["attention_type"] == "global"
    assert config["use_power_spectrum"] is True
    assert config["input_channels"] == 5
    assert config["psd_multiscale"] is False
    assert config["psd_multiscale_separate_channels"] is False
    assert config["psd_frequency_band_channels"] is True
    assert config["psd_frequency_bands"] == (
        (0.03, 0.05),
        (0.09, 0.11),
        (0.14, 0.16),
        (0.22, 0.24),
    )
