import pytest
import torch
from mrboost.computation import (
    fft_2D,
    fft_nD,
    generate_golden_angle_radial_spokes_kspace_trajectory,
    generate_golden_angle_stack_of_stars_kspace_trajectory,
    nufft_adj_2d,
    nufft_adj_3d,
)


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


# Test for generate_golden_angle_radial_spokes_kspace_trajectory
def test_generate_golden_angle_radial_spokes_kspace_trajectory():
    # Test case 1: Basic shape check
    spokes_num = 2
    spoke_length = 4
    ktraj = generate_golden_angle_radial_spokes_kspace_trajectory(
        spokes_num, spoke_length
    )
    assert ktraj.shape == (
        2,
        spokes_num,
        spoke_length,
    ), "Shape mismatch for 2D radial trajectory"

    traj_gt = torch.tensor(
        [
            [
                [-3.1416, -1.5708, 0.0000, 1.5708],  # kx for spoke 1
                [1.1384, 0.5692, -0.0000, -0.5692],
            ],  # kx for spoke 2
            [
                [0.0000, 0.0000, 0.0000, 0.0000],  # ky for spoke 1
                [-2.9281, -1.4640, 0.0000, 1.4640],
            ],  # ky for spoke 2
        ]
    )
    # Test case 2: Check that the center of the k-space is zero
    assert torch.allclose(
        ktraj, traj_gt, atol=1e-3, rtol=1e-3
    ), "Trajectory does not match expected values"


# Test for generate_golden_angle_stack_of_stars_kspace_trajectory
def test_generate_golden_angle_stack_of_stars_kspace_trajectory():
    # Test case 1: Basic shape check
    spokes_num = 2
    spoke_length = 4
    kz_mask = torch.tensor([1, 0, 1])
    ktraj_3d = generate_golden_angle_stack_of_stars_kspace_trajectory(
        spokes_num, spoke_length, kz_mask
    )
    kz_num_sampled = torch.sum(kz_mask).item()
    assert ktraj_3d.shape == (
        3,
        kz_num_sampled,
        spokes_num,
        spoke_length,
    ), "Shape mismatch for 3D stack-of-stars trajectory"

    # Test case 2: Check that the kz-axis positions are correctly applied
    kz_positions = 2 * torch.pi * torch.linspace(-0.5, 0.5, kz_mask.shape[-1])
    sampled_z_positions = kz_positions[kz_mask == 1]
    for i, z in enumerate(sampled_z_positions):
        assert torch.allclose(
            ktraj_3d[2, i, :, :],
            z * torch.ones(spokes_num, spoke_length),
            atol=1e-3,
            rtol=1e-3,
        ), "kz-axis positions are incorrect"

    # Test case 3: Check that the 2D trajectory is correctly repeated for each kz position
    ktraj_2d = generate_golden_angle_radial_spokes_kspace_trajectory(
        spokes_num, spoke_length
    )
    for i in range(kz_num_sampled):
        assert torch.allclose(
            ktraj_3d[0, i, :, :], ktraj_2d[0], atol=1e-6, rtol=1e-6
        ), "kx values are incorrect"
        assert torch.allclose(
            ktraj_3d[1, i, :, :], ktraj_2d[1], atol=1e-6, rtol=1e-6
        ), "ky values are incorrect"
