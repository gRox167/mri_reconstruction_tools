from numbers import Number
from types import NoneType
from typing import Sequence

import einx
import numpy as np
import scipy
import torch
import torch as ops
import torch.nn.functional as F
from jaxtyping import Shaped
from plum import dispatch, overload
from pytorch_finufft.functional import (
    FinufftType1,
    FinufftType2,
)
from torch import Tensor
from torch.fft import fft, fft2, fftn, fftshift, ifft, ifft2, ifftn, ifftshift
from tqdm import tqdm

# from .io_utils import *
from .type_utils import (
    ComplexImage2D,
    ComplexImage3D,
    KspaceData,
    KspaceSpokesData,
    KspaceSpokesTraj,
    KspaceSpokesTraj3D,
    KspaceTraj,
    KspaceTraj3D,
)

# else:
#     import numpy as np


def batch_process(batch_size: int, device: torch.device, batch_dim=0):
    def Inner(func):
        def process(*args, **kwargs):
            outputs = []
            kwargs_input = dict(
                (k, v.to(device)) if isinstance(v, torch.Tensor) else (k, v)
                for k, v in kwargs.items()
            )
            args_batched = [torch.split(data, batch_size, batch_dim) for data in args]
            print(args_batched[0][0].shape)
            batch_num = len(args_batched[0])
            for batch_idx in tqdm(range(batch_num)):
                args_input = (data[batch_idx].to(device) for data in args_batched)
                outputs.append(func(*args_input, **kwargs_input).cpu())
            outputs = torch.cat(outputs, dim=batch_dim)
            for k, v in kwargs_input.items():
                if isinstance(v, torch.Tensor):
                    v.cpu()
            return outputs

        return process

    return Inner


def hamming_filter(nonzero_width_percent: float, width: int) -> np.ndarray:
    nonzero_width = round(width * nonzero_width_percent)
    pad_width_L = round((width - nonzero_width) // 2)
    pad_width_R = width - nonzero_width - pad_width_L
    hamming_weights = np.float32(np.hamming(nonzero_width))
    W = np.pad(hamming_weights, pad_width=(pad_width_L, pad_width_R))
    return W


def hanning_filter(nonzero_width_percent: float, width: int) -> np.ndarray:
    nonzero_width = round(width * nonzero_width_percent)
    pad_width_L = round((width - nonzero_width) // 2)
    pad_width_R = width - nonzero_width - pad_width_L
    hanning_weights = np.float32(np.hanning(nonzero_width))
    print(pad_width_L, pad_width_R)
    W = np.pad(hanning_weights, pad_width=(pad_width_L, pad_width_R))
    return W


def tuned_and_robust_estimation(
    navigator: np.ndarray,
    percentW: float,
    Fs,
    FOV,
    ndata,
    device=torch.device("cuda"),
):
    """
    return channel and rotation index and generated curve
    """
    col_num, line_num, ch_num = navigator.shape

    # To reduce noise, the navigator k-space data were apodized using a Hamming window.
    W = hamming_filter(percentW / 100, col_num)
    W = einx.rearrange(
        "col_num -> col_num line_num ch_num",
        W,
        line_num=line_num,
        ch_num=ch_num,
    )

    # New quality metric block begin
    N = navigator.shape[1]
    f = torch.linspace(-0.5 * Fs, 0.5 * Fs - Fs / N, steps=N, device=device)
    # compute the ifft of weighted navigator, using the representation in CAPTURE paper
    # col_num->x, line_num->n, ch_num->i, tuning_num->m
    K_weighted = torch.from_numpy(W * navigator).to(f.device)
    projections = fftshift(
        ifft(ifftshift(K_weighted, dim=0), dim=0), dim=0
    )  # shape is x n i

    # shape is m=100
    phase_rotation_factors = torch.exp(
        -1j * 2 * torch.pi * torch.arange(1, 101, device=f.device) / 100
    )
    r = torch.empty((projections.shape[1], projections.shape[2], 100), device=f.device)
    for m in range(100):
        r[:, :, m] = torch.argmax(
            (phase_rotation_factors[m] * projections[:, :, :]).real, dim=0
        )
    # A = torch.einsum('xni,m->xnim',projections,phase_rotation_factors).real # np.multiply.outer(projections, phase_rorate..)
    # r = torch.argmax(A,dim=0).to(torch.double)+1 # 'x n i m -> n i m'
    R = torch.abs(fftshift(fft(r - einx.mean("n i m ->i m", r), dim=0), dim=0))

    lowfreq_integral = einx.sum(
        "f i m -> i m", R[(torch.abs(f) < 0.5) * (torch.abs(f) > 0.1)]
    )
    highfreq_integral = einx.sum("f i m -> i m", R[torch.abs(f) > 0.8])
    r_range = einx.max("n i m -> i m", r).values - einx.min("n i m -> i m", r).values
    lower_bound = torch.full_like(r_range, 30 / (FOV / (ndata / 2)))
    # what does this FOV/ndata use for
    determinator = torch.maximum(r_range, lower_bound)
    Q = lowfreq_integral / highfreq_integral / determinator
    Q_np = Q.numpy(force=True)  # faster than matlab version 10x

    i_max, m_max = np.unravel_index(np.argmax(Q_np), Q_np.shape)
    # projection_max = projections[:, :, i_max]
    r_max = r[:, i_max, m_max].numpy(force=True)
    # new quality metric block end

    # filter high frequency signal
    b = scipy.signal.firwin(12, 1 / (Fs / 2), window="hamming", pass_zero="lowpass")
    a = 1
    r_max_low_pass = scipy.signal.filtfilt(b, a, r_max)
    r_max_SG = scipy.signal.filtfilt(b, a, scipy.signal.savgol_filter(r_max, 5, 1))
    r_max_filtered = r_max_low_pass.copy()
    r_max_filtered[0:10], r_max_filtered[-10:] = r_max_SG[0:10], r_max_SG[-10:]

    return i_max, m_max, torch.from_numpy(r_max_filtered)


def centralize_kspace(
    kspace_data, acquire_length, center_idx_in_acquire_lenth, full_length, dim
) -> torch.Tensor:
    # center_in_acquire_length is index, here +1 to turn into quantity
    front_padding = round(full_length / 2 - (center_idx_in_acquire_lenth + 1))
    # the dc point can be located at length/2 or length/2+1, when length is even, cihat use length/2+1
    # front_padding += 1
    pad_length = [0 for i in range(2 * len(kspace_data.shape))]
    pad_length[dim * 2 + 1], pad_length[dim * 2] = (
        front_padding,
        full_length - acquire_length - front_padding,
    )
    pad_length.reverse()
    # torch.nn.functional.pad() are using pad_lenth in a inverse way.
    # (pad_front for axis -1,pad_back for axis -1, pad_front for axis -2, pad_back for axis-2 ......)
    kspace_data_mask = ops.ones(kspace_data.shape, dtype=ops.bool)
    kspace_data_mask = F.pad(kspace_data_mask, pad_length, mode="constant", value=False)
    kspace_data_ = F.pad(
        kspace_data, pad_length, mode="constant"
    )  # default constant is 0

    return kspace_data_, kspace_data_mask


def ifft_1D(kspace_data: Tensor, dim=-1, norm="ortho") -> Tensor:
    return fftshift(ifft(ifftshift(kspace_data, dim=dim), dim=dim, norm=norm), dim=dim)


def fft_1D(image_data: Tensor, dim=-1, norm="ortho") -> Tensor:
    return ifftshift(fft(fftshift(image_data, dim=dim), dim=dim, norm=norm), dim=dim)


def ifft_2D(kspace_data: Tensor, dim=(-2, -1), norm="ortho") -> Tensor:
    return fftshift(ifft2(ifftshift(kspace_data, dim=dim), dim=dim, norm=norm), dim=dim)


def fft_2D(image_data: Tensor, dim=(-2, -1), norm="ortho") -> Tensor:
    return ifftshift(fft2(fftshift(image_data, dim=dim), dim=dim, norm=norm), dim=dim)


def ifft_nD(kspace_data: Tensor, dim=(-3, -2, -1), norm="ortho") -> Tensor:
    return fftshift(ifftn(ifftshift(kspace_data, dim=dim), dim=dim, norm=norm), dim=dim)


def fft_nD(image_data: Tensor, dim=(-3, -2, -1), norm="ortho") -> Tensor:
    return ifftshift(fftn(fftshift(image_data, dim=dim), dim=dim, norm=norm), dim=dim)


def generate_golden_angle_radial_spokes_kspace_trajectory(spokes_num, spoke_length):
    """
    Generate a 2D radial k-space trajectory with a golden angle pattern.

    Args:
        spokes_num (int): Number of spokes in the 2D radial trajectory.
        spoke_length (int): Number of samples along each spoke.

    Returns:
        torch.Tensor: A 2D k-space trajectory of shape (2, spokes_num, spoke_length).
    """
    # Golden angle in radians
    KWIC_GOLDENANGLE = (np.sqrt(5) - 1) / 2 * np.pi  # Golden angle in radians

    # Create the k-space trajectory for each spoke
    # k = torch.linspace(-0.5, 0.5 - 1 / spoke_length, spoke_length)
    # k[spoke_length // 2] = 0
    # k = torch.linspace(-0.5, 0.5, spoke_length)
    k = torch.arange(spoke_length) / spoke_length - 0.5

    # Generate the angles for each spoke
    A = torch.arange(spokes_num) * KWIC_GOLDENANGLE

    # Calculate kx and ky for each spoke
    kx = torch.outer(torch.cos(A), k)
    ky = torch.outer(torch.sin(A), k)

    # Stack kx and ky to form the 2D k-space trajectory
    ktraj = torch.stack((kx, ky), dim=0)

    # Scale by 2*pi to match the k-space units
    return ktraj * 2 * np.pi


def generate_golden_angle_stack_of_stars_kspace_trajectory(
    spokes_num, spoke_length, kz_mask
):
    """
    Generate a 3D stack-of-stars k-space trajectory with a golden angle radial pattern.

    Args:
        spokes_num (int): Number of spokes in the 2D radial trajectory.
        spoke_length (int): Number of samples along each spoke.
        kz_mask (torch.Tensor): A binary mask of shape (slices_num,) indicating which kz-positions are sampled.
                                A value of 1 means the kz-position is sampled, and 0 means it is not.

    Returns:
        torch.Tensor: A 3D k-space trajectory of shape (3, kz_num, total_spokes_on_kxky, spoke_length).
    """
    # Generate the 2D radial spokes trajectory
    ktraj_2d = generate_golden_angle_radial_spokes_kspace_trajectory(
        spokes_num, spoke_length
    )
    kz_num = kz_mask.shape[-1]
    # Generate the kz-axis positions
    kz_positions = 2 * torch.pi * (torch.arange(kz_num) / kz_num - 0.5)
    # kz_positions = 2 * torch.pi * torch.linspace(-0.5, 0.5, kz_num)

    # Apply the kz_mask to select which z-positions are sampled
    sampled_z_positions = kz_positions[kz_mask == 1]

    ktraj_3d = einx.rearrange(
        "v sp len, kz -> (v + 1) kz sp len", ktraj_2d, sampled_z_positions
    )
    return ktraj_3d


def data_binning(
    data,
    sorted_r_idx,
    contrast_num,
    spokes_per_contra,
    phase_num,
    spokes_per_phase,
):
    spoke_len = data.shape[-1]

    output = einx.get_at(  # sp_t=n
        "... (t [sp_t]) spoke_len, t n -> ... t n spoke_len",
        data,
        sorted_r_idx,
        t=contrast_num,
        # sp_t=spokes_per_contra,
        spoke_len=spoke_len,
    )
    return einx.rearrange(
        "... t (ph spoke) spoke_len -> t ph ...  spoke spoke_len",
        output,
        ph=phase_num,
        spoke=spokes_per_phase,
    )

    # output = einx.rearrange(
    #     "... (t spokes_per_contra) spoke_len -> ... t spokes_per_contra spoke_len ",
    #     data,
    #     t=contrast_num,
    #     spokes_per_contra=spokes_per_contra,
    # )
    # output = output.gather(
    #     dim=-2,
    #     index=einx.repeat(
    #         "t spokes_per_contra -> t spokes_per_contra spoke_len",
    #         sorted_r_idx,
    #         spokes_per_contra=spokes_per_contra,
    #         spoke_len=spoke_len,
    #     ).expand_as(output),
    # )
    # output = rearrange(
    #     output,
    #     "... t (ph spoke) spoke_len -> t ph ...  spoke spoke_len",
    #     ph=phase_num,
    #     spoke=spokes_per_phase,
    # )
    # return output


def data_binning_phase(data, sorted_r_idx, phase_num, spokes_per_phase):
    spoke_len = data.shape[-1]
    return einx.get_at(
        "... ([ph spoke]) spoke_len, ([ph spoke]) -> ph ... spoke spoke_len",
        data,
        sorted_r_idx,
        ph=phase_num,
        spoke=spokes_per_phase,
        spoke_len=spoke_len,
    )

    # output = data.gather(
    #     dim=-2,
    #     index=repeat(
    #         sorted_r_idx, "spoke -> spoke spoke_len", spoke_len=spoke_len
    #     ).expand_as(data),
    # )
    # output = rearrange(
    #     output,
    #     "...  (ph spoke) spoke_len -> ph ...  spoke spoke_len",
    #     ph=phase_num,
    #     spoke=spokes_per_phase,
    # )
    # return output


def data_binning_consecutive(data, spokes_per_contra):
    assert data.shape[-2] % spokes_per_contra == 0
    output = einx.rearrange(
        "... (t spokes_per_contra) spoke_len -> t ... spokes_per_contra spoke_len",
        data,
        spokes_per_contra=spokes_per_contra,
    )
    return output


# def recon_adjnufft(kspace_data, kspace_traj, kspace_density_compensation, adjnufft_ob):
#     return adjnufft_ob(
#         rearrange(
#             kspace_data * kspace_density_compensation,
#             "... spoke spoke_len-> ... (spoke spoke_len)",
#         ),
#         rearrange(kspace_traj, "complx spoke spoke_len -> complx (spoke spoke_len)"),
#     )


def polygon_area(vertices):
    """
    vertice are tensor, vertices_num x dimensions(2)
    """
    x, y = vertices[:, 0], vertices[:, 1]
    correction = x[-1] * y[0] - y[-1] * x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5 * np.abs(main_area + correction)


def normalization(img):
    return (img - img.mean()) / img.std()


def normalization_root_of_sum_of_square(d, dim=0):
    ndims = len(d.shape)
    dim_to_reduce = tuple([i for i in range(ndims) if i != dim])
    k = ops.sqrt(ops.sum(d * d.conj(), dim=dim_to_reduce, keepdim=True))
    # let the average of energy in each ksample point be 1
    # print(k)
    # print((d**2).mean())
    # print(((d/k)**2).mean())
    return d / k


@overload
def nufft_2d(
    images: Shaped[ComplexImage2D, "*channel"],
    kspace_traj: KspaceTraj,
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[KspaceData, " *channel"]:
    if norm_factor is None:
        norm_factor = np.sqrt(np.prod(image_size))

    return (
        FinufftType2.apply(
            kspace_traj.flip(0),
            images,
            dict(isign=-1, modeord=0),
        )
        / norm_factor
    )


@overload
def nufft_2d(
    images: Shaped[ComplexImage2D, "..."],
    kspace_traj: Shaped[KspaceTraj, "... batch"],
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[KspaceData, "..."]:
    *batch_shape, _, length = kspace_traj.shape
    batch_size = np.prod(batch_shape, dtype=int)
    kspace_traj_batched = kspace_traj.view(-1, 2, length)

    *channel_shape, h, w = images.shape[len(batch_shape) :]
    images_batched = images.view(batch_size, *channel_shape, h, w)

    output = torch.stack(
        [
            nufft_2d(
                images_batched[i],
                kspace_traj_batched[i],
                image_size,
                norm_factor,
            )
            for i in range(batch_size)
        ],
    )
    return output.view(*batch_shape, *channel_shape, length)


@dispatch
def nufft_2d(
    images,
    kspace_traj,
    image_size,
    norm_factor,
):
    pass


@overload
def nufft_3d(
    images: Shaped[ComplexImage3D, "*channel"],
    kspace_traj: KspaceTraj3D,
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[KspaceData, " *channel"]:
    if norm_factor is None:
        norm_factor = np.sqrt(np.prod(image_size))
    return (
        FinufftType2.apply(
            kspace_traj.flip(0),
            images,
            dict(isign=-1, modeord=0),
        )
        / norm_factor
    )


@overload
def nufft_3d(
    images: Shaped[ComplexImage3D, "..."],
    kspace_traj: Shaped[KspaceTraj3D, "... batch"],
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[KspaceData, "..."]:
    *batch_shape, _, length = kspace_traj.shape
    batch_size = np.prod(batch_shape, dtype=int)
    kspace_traj_batched = kspace_traj.view(-1, 3, length)

    *channel_shape, d, h, w = images.shape[len(batch_shape) :]
    images_batched = images.view(batch_size, *channel_shape, d, h, w)

    output = torch.stack(
        [
            nufft_3d(
                images_batched[i],
                kspace_traj_batched[i],
                image_size,
                norm_factor,
            )
            for i in range(batch_size)
        ],
    )
    return output.view(*batch_shape, *channel_shape, length)


@dispatch
def nufft_3d(
    images,
    kspace_traj,
    image_size,
    norm_factor,
):
    pass


@overload
def nufft_adj_2d(
    kspace_data: Shaped[KspaceData, "*channel"],
    kspace_traj: KspaceTraj,
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[ComplexImage2D, "*channel"]:
    if norm_factor is None:
        norm_factor = np.sqrt(np.prod(image_size))
    return (
        FinufftType1.apply(
            kspace_traj.flip(0),
            kspace_data,
            tuple(image_size),
            dict(isign=1, modeord=0),
        )
        / norm_factor
    )


@overload
def nufft_adj_2d(
    kspace_data: Shaped[KspaceData, "..."],
    kspace_traj: Shaped[KspaceTraj, "... batch"],
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[ComplexImage2D, "..."]:
    *batch_shape, _, length = kspace_traj.shape
    batch_size = np.prod(batch_shape, dtype=int)

    kspace_traj_batched = einx.rearrange("... comp len -> (...) comp len", kspace_traj)

    *channel_shape, length = kspace_data.shape[len(batch_shape) :]
    kspace_data_batched = kspace_data.view(batch_size, *channel_shape, length)
    output = torch.stack(
        [
            nufft_adj_2d(
                kspace_data_batched[i],
                kspace_traj_batched[i],
                tuple(image_size),
                norm_factor,
            )
            for i in range(batch_size)
        ],
    )
    # return einx.rearrange("(b...) ch... h w -> b... ch... h w", output, b=batch_shape)
    return output.view(*batch_shape, *channel_shape, *image_size)


@dispatch
def nufft_adj_2d(
    kspace_data,
    kspace_traj,
    image_size,
    norm_factor,
):
    pass


@overload
def nufft_adj_3d(
    kspace_data: Shaped[KspaceData, "*channel"],
    kspace_traj: KspaceTraj3D,
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[ComplexImage3D, "*channel"]:
    if norm_factor is None:
        norm_factor = np.sqrt(np.prod(image_size))
    return (
        FinufftType1.apply(
            kspace_traj.flip(0),  # kx ky kz -> kz ky kx corresponds to d h w
            kspace_data,
            tuple(image_size),
            dict(isign=1, modeord=0),
        )
        / norm_factor
    )


@overload
def nufft_adj_3d(
    kspace_data: Shaped[KspaceData, "..."],
    kspace_traj: Shaped[KspaceTraj3D, "... batch"],
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[ComplexImage3D, "..."]:
    *batch_shape, _, length = kspace_traj.shape
    batch_size = np.prod(batch_shape, dtype=int)

    kspace_traj_batched = einx.rearrange("... v len -> (...) v len", kspace_traj)

    *channel_shape, length = kspace_data.shape[len(batch_shape) :]
    kspace_data_batched = kspace_data.view(batch_size, *channel_shape, length)
    output = torch.stack(
        [
            nufft_adj_3d(
                kspace_data_batched[i],
                kspace_traj_batched[i],
                tuple(image_size),
                norm_factor,
            )
            for i in range(batch_size)
        ],
    )
    # return einx.rearrange(
    #     "(b...) ch... d h w -> b... ch... d h w", output, b=batch_shape
    # )
    return output.view(*batch_shape, *channel_shape, *image_size)


@dispatch
def nufft_adj_3d(
    kspace_data,
    kspace_traj,
    image_size,
    norm_factor,
):
    pass


def radial_spokes_to_kspace_point(
    x: Shaped[KspaceSpokesData, "..."]
    | Shaped[KspaceSpokesTraj, "..."]
    | Shaped[KspaceSpokesTraj3D, "..."],
):
    return einx.rearrange(
        "... middle len -> ... (middle len)",
        x,
    )


def kspace_point_to_radial_spokes(
    x: Shaped[KspaceData, "..."] | Shaped[KspaceSpokesTraj3D, "..."],
    spoke_len: int,
):
    return einx.rearrange(
        "... (middle len) -> ... middle len",
        x,
        len=spoke_len,
    )
