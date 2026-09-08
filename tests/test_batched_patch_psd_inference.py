import numpy as np
import torch

from models.predict import predict_bad_regions_probability
from utils.image_utils import (
    compute_frequency_band_psd_channels,
    compute_frequency_band_psd_channels_batch,
    compute_multiscale_radial_psd_image,
    compute_multiscale_radial_psd_image_batch,
)


def test_multiscale_radial_psd_batch_matches_single_patch_loop():
    rng = np.random.default_rng(123)
    images = rng.normal(size=(4, 48, 48)).astype(np.float32)

    single = np.stack(
        [
            compute_multiscale_radial_psd_image(
                image,
                scales=(8, 16, 32),
                normalize=True,
            )
            for image in images
        ],
        axis=0,
    )
    batched = compute_multiscale_radial_psd_image_batch(
        images,
        scales=(8, 16, 32),
        normalize=True,
    )

    assert np.allclose(single, batched, atol=1e-6)


def test_batched_patch_psd_prediction_matches_sequential_path():
    class EchoPsdModel(torch.nn.Module):
        def forward(self, x):
            return (x[:, 0:1] + 0.25 * x[:, 1:2]).float()

    rng = np.random.default_rng(321)
    image = rng.normal(size=(72, 80)).astype(np.float32)
    model = EchoPsdModel().eval()
    kwargs = dict(
        model=model,
        image=image,
        device="cpu",
        patch_size=48,
        overlap=24,
        pixel_size_angstrom=2.0,
        use_power_spectrum=True,
        include_real_space_input=True,
        use_hann_blending=True,
        blending_window="tukey",
        blending_edge_px=8,
        psd_multiscale=True,
        psd_multiscale_separate_channels=False,
        multiscale_psd_source="patch",
        psd_scales=(8, 16, 32),
        normalization_method="percentile",
    )

    sequential = predict_bad_regions_probability(batch_forward_size=1, **kwargs)
    batched = predict_bad_regions_probability(batch_forward_size=4, **kwargs)

    assert np.allclose(sequential, batched, atol=1e-6)


def test_frequency_band_psd_batch_matches_single_patch_loop():
    rng = np.random.default_rng(456)
    images = rng.normal(size=(3, 64, 64)).astype(np.float32)
    bands = ((0.03, 0.05), (0.09, 0.11), (0.14, 0.16), (0.22, 0.24))

    single = np.stack(
        [
            compute_frequency_band_psd_channels(
                image,
                pixel_size_angstrom=2.0,
                frequency_bands=bands,
                normalize=True,
                use_radial_normalization=True,
            )
            for image in images
        ],
        axis=0,
    )
    batched = compute_frequency_band_psd_channels_batch(
        images,
        pixel_size_angstrom=2.0,
        frequency_bands=bands,
        normalize=True,
        use_radial_normalization=True,
    )

    assert np.allclose(single, batched, atol=1e-6)
