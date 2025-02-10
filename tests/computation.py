import pytest
import torch
from mrboost.computation import fft_2D, fft_nD, nufft_adj_2d, nufft_adj_3d


@pytest.fixture
def single_channel_3d_data():
    kspace_data = torch.rand(128, dtype=torch.complex64)
    kspace_traj = torch.rand(3, 128)
    image_size = (64, 64, 64)
    return kspace_data, kspace_traj, image_size


@pytest.fixture
def multi_channel_3d_data():
    kspace_data = torch.rand(4, 128, dtype=torch.complex64)
    kspace_traj = torch.rand(3, 128)
    image_size = (64, 64, 64)
    return kspace_data, kspace_traj, image_size


@pytest.fixture
def batched_3d_data():
    kspace_data = torch.rand(2, 4, 128, dtype=torch.complex64)
    kspace_traj = torch.rand(2, 3, 128)
    image_size = (64, 64, 64)
    return kspace_data, kspace_traj, image_size


@pytest.fixture
def single_channel_2d_data():
    kspace_data = torch.rand(128, dtype=torch.complex64)
    kspace_traj = torch.rand(2, 128)
    image_size = (64, 64)
    return kspace_data, kspace_traj, image_size


@pytest.fixture
def multi_channel_2d_data():
    kspace_data = torch.rand(4, 128, dtype=torch.complex64)
    kspace_traj = torch.rand(2, 128)
    image_size = (64, 64)
    return kspace_data, kspace_traj, image_size


@pytest.fixture
def batched_2d_data():
    kspace_data = torch.rand(2, 4, 128, dtype=torch.complex64)
    kspace_traj = torch.rand(2, 2, 128)
    image_size = (64, 64)
    return kspace_data, kspace_traj, image_size


@pytest.fixture
def known_3d_image_and_kspace():
    image_size = (32, 64, 128)
    image = torch.zeros(image_size, dtype=torch.complex64)
    image[8:24, 16:48, 32:96] = 1.0
    kspace_data = fft_nD(image, dim=(-3, -2, -1), norm="ortho")
    kx, ky, kz = torch.meshgrid(
        torch.linspace(-0.5, 0.5, image_size[0]) * 2 * torch.pi,
        torch.linspace(-0.5, 0.5, image_size[1]) * 2 * torch.pi,
        torch.linspace(-0.5, 0.5, image_size[2]) * 2 * torch.pi,
        indexing="ij",
    )
    kspace_traj = torch.stack([kx.flatten(), ky.flatten(), kz.flatten()], dim=0)
    kspace_data = kspace_data.flatten()
    return image, kspace_data, kspace_traj, image_size


@pytest.fixture
def known_2d_image_and_kspace():
    image_size = (128, 128)
    image = torch.zeros(image_size, dtype=torch.complex64)
    image[32:96, 32:96] = 1.0
    kspace_data = fft_2D(image, dim=(-2, -1), norm="ortho")
    kx, ky = torch.meshgrid(
        torch.linspace(-0.5, 0.5, image_size[0]) * 2 * torch.pi,
        torch.linspace(-0.5, 0.5, image_size[1]) * 2 * torch.pi,
        indexing="ij",
    )
    kspace_traj = torch.stack([kx.flatten(), ky.flatten()], dim=0)
    kspace_data = kspace_data.flatten()
    return image, kspace_data, kspace_traj, image_size


def test_nufft_adj_3d_single_channel(single_channel_3d_data):
    kspace_data, kspace_traj, image_size = single_channel_3d_data
    result = nufft_adj_3d(kspace_data, kspace_traj, image_size)
    assert result.shape == (64, 64, 64)
    assert torch.is_complex(result)


def test_nufft_adj_3d_multi_channel(multi_channel_3d_data):
    kspace_data, kspace_traj, image_size = multi_channel_3d_data
    result = nufft_adj_3d(kspace_data, kspace_traj, image_size)
    assert result.shape == (4, 64, 64, 64)
    assert torch.is_complex(result)


def test_nufft_adj_3d_batched(batched_3d_data):
    kspace_data, kspace_traj, image_size = batched_3d_data
    result = nufft_adj_3d(kspace_data, kspace_traj, image_size)
    assert result.shape == (2, 4, 64, 64, 64)
    assert torch.is_complex(result)


def test_nufft_adj_3d_with_norm_factor(single_channel_3d_data):
    kspace_data, kspace_traj, image_size = single_channel_3d_data
    norm_factor = 10.0
    result = nufft_adj_3d(kspace_data, kspace_traj, image_size, norm_factor)
    assert result.shape == (64, 64, 64)
    assert torch.is_complex(result)


def test_nufft_adj_3d_correctness(known_3d_image_and_kspace):
    image, kspace_data, kspace_traj, image_size = known_3d_image_and_kspace
    reconstructed_image = nufft_adj_3d(kspace_data, kspace_traj, image_size)
    assert torch.allclose(reconstructed_image.abs(), image.abs(), atol=5e-1, rtol=1e-1)


def test_nufft_adj_2d_single_channel(single_channel_2d_data):
    kspace_data, kspace_traj, image_size = single_channel_2d_data
    result = nufft_adj_2d(kspace_data, kspace_traj, image_size)
    assert result.shape == (64, 64)
    assert torch.is_complex(result)


def test_nufft_adj_2d_multi_channel(multi_channel_2d_data):
    kspace_data, kspace_traj, image_size = multi_channel_2d_data
    result = nufft_adj_2d(kspace_data, kspace_traj, image_size)
    assert result.shape == (4, 64, 64)
    assert torch.is_complex(result)


def test_nufft_adj_2d_batched(batched_2d_data):
    kspace_data, kspace_traj, image_size = batched_2d_data
    result = nufft_adj_2d(kspace_data, kspace_traj, image_size)
    assert result.shape == (2, 4, 64, 64)
    assert torch.is_complex(result)


def test_nufft_adj_2d_correctness(known_2d_image_and_kspace):
    image, kspace_data, kspace_traj, image_size = known_2d_image_and_kspace
    reconstructed_image = nufft_adj_2d(kspace_data, kspace_traj, image_size)
    assert torch.allclose(reconstructed_image.abs(), image.abs(), atol=5e-1, rtol=1e-1)
