from numbers import Number
from types import NoneType
from typing import Sequence

import einx

# import numpy as np
import scipy
import torch
import torch.nn.functional as F

# from juliacall import Main as jl
# jl.include("/data-local/anlab/Chunxu/mri_reconstruction_tools/src/mrboost/computation.jl")
from einops import rearrange, reduce, repeat

from icecream import ic
from jax import numpy as np
from jaxtyping import Shaped
from plum import dispatch, overload
from pytorch_finufft.functional import (
    FinufftType1,
    FinufftType2,
    finufft_type1,
    finufft_type2,
)
from torch.fft import fft, fftshift, ifft, ifftshift
from tqdm import tqdm

from .io_utils import *
from .type_utils import (
    ComplexImage2D,
    KspaceData,
    KspaceSpokesData,
    KspaceSpokesTraj,
    KspaceTraj,
)
import concurrent.futures

def batch_process(batch_size: int, device: torch.device, batch_dim=0):
    def Inner(func):
        def process(*args, **kwargs):
            outputs = []
            kwargs_input = dict(
                (k, v.to(device)) if isinstance(v, torch.Tensor) else (k, v)
                for k, v in kwargs.items()
            )
            args_batched = [
                torch.split(data, batch_size, batch_dim) for data in args
            ]
            print(args_batched[0][0].shape)
            batch_num = len(args_batched[0])
            for batch_idx in tqdm(range(batch_num)):
                args_input = (
                    data[batch_idx].to(device) for data in args_batched
                )
                outputs.append(func(*args_input, **kwargs_input).cpu())
            outputs = torch.cat(outputs, dim=batch_dim)
            for k, v in kwargs_input.items():
                if isinstance(v, torch.Tensor):
                    v.cpu()
            return outputs

        return process

    return Inner


def hamming_filter(nonzero_width_percent: float, width: int) -> np.ndarray:
    nonzero_width = round(width * nonzero_width_percent) #640*0,05=32
    pad_width_L = round((width - nonzero_width) // 2) # (640-32)/2=304
    pad_width_R = width - nonzero_width - pad_width_L # 640-32-304=304
    hamming_weights = np.float32(np.hamming(nonzero_width))
    W = np.pad(hamming_weights, pad_width=(pad_width_L, pad_width_R))
    return W

def hanning_filter(nonzero_width_percent: float, width: int) -> np.ndarray:
    nonzero_width = round(width * nonzero_width_percent) #640*0,05=32
    pad_width_L = round((width - nonzero_width) // 2) # (640-32)/2=304
    pad_width_R = width - nonzero_width - pad_width_L # 640-32-304=304
    hamming_weights = np.float32(np.hanning(nonzero_width))
    W = np.pad(hamming_weights, pad_width=(pad_width_L, pad_width_R))
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
    W = repeat(
        W,
        "col_num -> col_num line_num ch_num",
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
    r = torch.empty(
        (projections.shape[1], projections.shape[2], 100), device=f.device
    )
    for m in range(100):
        r[:, :, m] = torch.argmax(
            (phase_rotation_factors[m] * projections[:, :, :]).real, dim=0
        )
    # A = torch.einsum('xni,m->xnim',projections,phase_rotation_factors).real # np.multiply.outer(projections, phase_rorate..)
    # r = torch.argmax(A,dim=0).to(torch.double)+1 # 'x n i m -> n i m'
    R = torch.abs(
        fftshift(fft(r - reduce(r, "n i m -> i m", "mean"), dim=0), dim=0)
    )

    lowfreq_integral = reduce(
        R[(torch.abs(f) < 0.5) * (torch.abs(f) > 0.1)], "f i m -> i m", "sum"
    )
    highfreq_integral = reduce(R[torch.abs(f) > 0.8], "f i m -> i m", "sum")
    r_range = reduce(r, "n i m -> i m", "max") - reduce(
        r, "n i m -> i m", "min"
    )
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
    b = scipy.signal.firwin(
        12, 1 / (Fs / 2), window="hamming", pass_zero="lowpass"
    )
    a = 1
    r_max_low_pass = scipy.signal.filtfilt(b, a, r_max)
    r_max_SG = scipy.signal.filtfilt(
        b, a, scipy.signal.savgol_filter(r_max, 5, 1)
    )
    r_max_filtered = r_max_low_pass.copy()
    r_max_filtered[0:10], r_max_filtered[-10:] = r_max_SG[0:10], r_max_SG[-10:]

    return i_max, m_max, torch.from_numpy(r_max_filtered)


def centralize_kspace(
    kspace_data, acquire_length, center_idx_in_acquire_lenth, full_length, dim
) -> torch.Tensor:
    # center_in_acquire_length is index, here +1 to turn into quantity
    front_padding = round(full_length / 2 - (center_idx_in_acquire_lenth + 1))
    # the dc point can be located at length/2 or length/2+1, when length is even, cihat use length/2+1
    front_padding += 1
    pad_length = [0 for i in range(2 * len(kspace_data.shape))]
    pad_length[dim * 2 + 1], pad_length[dim * 2] = (
        front_padding,
        full_length - acquire_length - front_padding,
    )
    pad_length.reverse()
    # torch.nn.functional.pad() are using pad_lenth in a inverse way.
    # (pad_front for axis -1,pad_back for axis -1, pad_front for axis -2, pad_back for axis-2 ......)
    kspace_data_mask = torch.ones(kspace_data.shape, dtype=torch.bool)
    kspace_data_mask = F.pad(
        kspace_data_mask, pad_length, mode="constant", value=False
    )
    kspace_data_ = F.pad(
        kspace_data, pad_length, mode="constant"
    )  # default constant is 0

    return kspace_data_, kspace_data_mask


def ifft_1D(kspace_data, dim=-1, norm="ortho"):
    return fftshift(
        ifft(ifftshift(kspace_data, dim=dim), dim=dim, norm=norm), dim=dim
    )


def fft_1D(image_data, dim=-1, norm="ortho"):
    return ifftshift(
        fft(fftshift(image_data, dim=dim), dim=dim, norm=norm), dim=dim
    )


def generate_golden_angle_radial_spokes_kspace_trajectory(
    spokes_num, spoke_length
):
    # create a k-space trajectory
    KWIC_GOLDENANGLE = (np.sqrt(5) - 1) / 2  # radian *180 = 111.246117975
    k = torch.linspace(-0.5, 0.5 - 1 / spoke_length, spoke_length) #[start:end:step] create one-d array start from -0.5,and contains 640 steps.
    k[spoke_length // 2] = 0 # make the center point to be 0
    A = torch.arange(spokes_num) * torch.pi * KWIC_GOLDENANGLE  # /180 #create the angle of each spoke
    kx = torch.outer(torch.cos(A), k) #outer product: cos(A) by k, #sampling cosine function
    ky = torch.outer(torch.sin(A), k)
    ktraj = torch.stack((kx, ky), dim=0)
    # ktraj = torch.complex(kx, ky)
    return ktraj * 2 * torch.pi 


def data_binning(
    data,
    sorted_r_idx,
    contrast_num,
    spokes_per_contra,
    phase_num,
    spokes_per_phase,
):
    spoke_len = data.shape[-1]
    output = rearrange(
        data,
        "... (t spokes_per_contra) spoke_len -> ... t spokes_per_contra spoke_len ",
        t=contrast_num,
        spokes_per_contra=spokes_per_contra,
    )
    # print(output.shape, sorted_r_idx.shape)
    output = output.gather(
        dim=-2,
        index=repeat(
            sorted_r_idx,
            "t spokes_per_contra -> t spokes_per_contra spoke_len",
            spokes_per_contra=spokes_per_contra,
            spoke_len=spoke_len,
        ).expand_as(output),
    )
    output = rearrange(
        output,
        "... t (ph spoke) spoke_len -> t ph ...  spoke spoke_len",
        ph=phase_num,
        spoke=spokes_per_phase,
    )
    return output


def data_binning_phase(data, sorted_r_idx, phase_num, spokes_per_phase):
    spoke_len = data.shape[-1]
    output = data.gather(
        dim=-2,
        index=repeat(
            sorted_r_idx, "spoke -> spoke spoke_len", spoke_len=spoke_len
        ).expand_as(data),
    )
    output = rearrange(
        output,
        "...  (ph spoke) spoke_len -> ph ...  spoke spoke_len",
        ph=phase_num,
        spoke=spokes_per_phase,
    )
    return output


def data_binning_consecutive(data, spokes_per_contra):
    assert data.shape[-2] % spokes_per_contra == 0
    output = rearrange(
        data,
        "... (t spokes_per_contra) spoke_len -> t ... spokes_per_contra spoke_len",
        spokes_per_contra=spokes_per_contra,
    )
    return output


# def data_binning_jl(data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase):
#     jl.GC.enable(False)
#     jl_output =  jl.data_binning(data.numpy(), sorted_r_idx.numpy(), contrast_num, spokes_per_contra, phase_num, spokes_per_phase)
#     output = jl_output.to_numpy()
#     jl.GC.enable(True)
#     return torch.from_numpy(output)


def recon_adjnufft(
    kspace_data, kspace_traj, kspace_density_compensation, adjnufft_ob
):
    return adjnufft_ob(
        rearrange(
            kspace_data * kspace_density_compensation,
            "... spoke spoke_len-> ... (spoke spoke_len)",
        ),
        rearrange(
            kspace_traj, "complx spoke spoke_len -> complx (spoke spoke_len)"
        ),
    )


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
    k = torch.sqrt(torch.sum(d * d.conj(), dim=dim_to_reduce, keepdim=True))
    # let the average of energy in each ksample point be 1
    # print(k)
    # print((d**2).mean())
    # print(((d/k)**2).mean())
    return d / k


def generate_nufft_op(image_size):
    norm_factor = np.sqrt(np.prod(image_size) * (2 ** len(image_size)))
    nufft_op = (
        lambda images, points: finufft_type2(
            points,
            images,
            isign=-1,
            modeord=0,
        )
        / norm_factor
    )
    nufft_adj_op = (
        lambda values, points: finufft_type1(
            points,
            values,
            tuple(image_size),
            isign=1,
            modeord=0,
        )
        / norm_factor
    )
    return nufft_op, nufft_adj_op


@overload
def nufft_2d(
    images: Shaped[ComplexImage2D, "*channel"],# *channel h w 
    kspace_traj: KspaceTraj, # 2 length
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[KspaceData, " *channel"]:
    if norm_factor is None:
        norm_factor = np.sqrt(np.prod(image_size))
    return (
        FinufftType2.apply(
            kspace_traj,
            images,
            dict(isign=-1, modeord=0),
        )
        / norm_factor 
    )


def nufft_2d_batched(
    images: Shaped[ComplexImage2D, "..."],
    kspace_traj: Shaped[KspaceTraj, "... batch"],
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
    max_workers: int = 50,
) -> Shaped[KspaceData, "..."]:
    """
    Parallel calls to `nufft_adj_2d` for each item in the batch, using threads.
    Returns a stacked output of shape [B, ...].
    
    Args:
      kspace_data_batched: batched data, shape [B, ...], each slice is for one sample
      kspace_traj_batched: shape [B, 2, ...] or [B, ...], each slice for one sample
      image_size: (H, W)
      norm_factor: normalization factor
      max_workers: how many worker threads to run in parallel
    """

    *batch_shape, _, length = kspace_traj.shape
    batch_size = np.prod(batch_shape, dtype=int)
    kspace_traj_batched = kspace_traj.view(-1, 2, length)

    *channel_shape, h, w = images.shape[len(batch_shape) :]
    images_batched = images.view(batch_size, *channel_shape, h, w)
    outputs = [None] * batch_size

    def worker(i: int):
        # Each thread processes one slice:
        return nufft_adj_2d(
            images_batched[i],
            kspace_traj_batched[i],
            image_size,
            norm_factor,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(batch_size)]
        for i, f in enumerate(futures):
            outputs[i] = f.result()

    # Stack the results
    return torch.stack(outputs, dim=0)

# def nufft_2d_loss(
#     images: Shaped[ComplexImage2D, "..."],
#     kspace_traj: Shaped[KspaceTraj, "... batch"],
#     image_size: Sequence[int],
#     norm_factor: Number | NoneType = None,
# ) -> Shaped[KspaceData, "..."]:
#     *batch_shape, _, length = kspace_traj.shape
#     batch_size = np.prod(batch_shape, dtype=int)
#     kspace_traj_batched = kspace_traj.view(-1, 2, length)

#     *channel_shape, h, w = images.shape[len(batch_shape) :]
#     images = einx.rearrange("b ch z h w -> b z ch h w", images)
#     images_batched = images.view(batch_size, *channel_shape, h, w)

#     output = torch.stack(
#         [
#             nufft_2d(
#                 images_batched[i],
#                 kspace_traj_batched[i],
#                 image_size,
#                 norm_factor,
#             )
#             for i in range(batch_size)
#         ],
#     )
#     return output.view(*batch_shape, *channel_shape, length)
        

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
    images_batched = images.contiguous().view(batch_size, *channel_shape, h, w) # 160, 320,320
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


def nufft_adj_2d_Inner(
    kspace_data: Shaped[KspaceData, "*channel"], # *channel length
    kspace_traj: KspaceTraj, # 2, sp*leng, 
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[ComplexImage2D, "*channel"]:
    # ic(kspace_data.shape)
    if norm_factor is None:
        norm_factor = np.sqrt(np.prod(image_size))
    return (
        FinufftType1.apply(
            kspace_traj,
            kspace_data,
            tuple(image_size),
            dict(isign=1, modeord=0),
        )
        / norm_factor
    )



############# training #################
@overload
def nufft_adj_2d(
    kspace_data: Shaped[KspaceData, "z ch"],# ch z leng
    kspace_traj: Shaped[KspaceTraj, "z ch"], # z ch 2 length
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[ComplexImage2D, "..."]:
    ic(kspace_traj.shape)# 5, 48,2 length
    ic(kspace_data.shape)
    *batch_shape, _, length = kspace_traj.shape # * batch_shape = z, ch; _ = 2, length = length
    batch_size = np.prod(batch_shape, dtype=int)
    ic(batch_size) # 160
    kspace_traj_batched = einx.rearrange(
        "b... comp len -> (b...) comp len", kspace_traj
    )
    
    *channel_shape, length = kspace_data.shape[len(batch_shape) :] # kspace_data.shape[2 :]
    # kspace_data = einx.rearrange("ch z len -> z ch len", kspace_data)
    kspace_data_batched = kspace_data.contiguous().view(batch_size, *channel_shape, length)
    ic(kspace_data_batched.shape)
    output = torch.stack(
        [
            nufft_adj_2d_Inner(
                kspace_data_batched[i],
                kspace_traj_batched[i], ### only loop slice dimension 
                tuple(image_size),
                norm_factor,
            )
            for i in range(batch_size)
        ],
    )
    return einx.rearrange(
        "(b...) ch... h w -> b... ch... h w", output, b=batch_shape
    )
    # return output.view(*batch_shape, *channel_shape, *image_size)

@overload
def nufft_adj_2d(
    kspace_data: Shaped[KspaceData, "ch z"],# ch z leng
    kspace_traj: KspaceTraj, # z ch 2 length
    image_size: Sequence[int],
    norm_factor: Number | NoneType = None,
) -> Shaped[ComplexImage2D, "..."]:
    ic(kspace_traj.shape)# 5, 48,2 length
    ic(kspace_data.shape)
    *batch_shape, _, length = kspace_traj.shape # * batch_shape = z, ch; _ = 2, length = length
    batch_size = np.prod(batch_shape, dtype=int)
    ic(batch_size) # 160
    kspace_traj_batched = einx.rearrange(
        "b... comp len -> (b...) comp len", kspace_traj
    )
    
    *channel_shape, length = kspace_data.shape[len(batch_shape) :] # kspace_data.shape[2 :]
    # kspace_data = einx.rearrange("ch z len -> z ch len", kspace_data)
    kspace_data_batched = kspace_data.contiguous().view(batch_size, *channel_shape, length)
    ic(kspace_data_batched.shape)
    output = torch.stack(
        [
            nufft_adj_2d_Inner(
                kspace_data_batched[i],
                kspace_traj_batched[i], ### only loop slice dimension 
                tuple(image_size),
                norm_factor,
            )
            for i in range(batch_size)
        ],
    )
    return einx.rearrange(
        "(b...) ch... h w -> b... ch... h w", output, b=batch_shape
    )

@dispatch
def nufft_adj_2d(
    kspace_data,
    kspace_traj,
    image_size,
    norm_factor,
):
    pass

########### testing ############
def nufft_adj_2d_batched(
    kspace_data: torch.Tensor,  # shape: [B, ...]
    kspace_traj: torch.Tensor,  # shape: [B, 2, num_points], etc.
    image_size: Sequence[int],
    norm_factor: Number | NoneType= None,
    max_workers: int = 50,
) -> torch.Tensor:
    """
    Parallel calls to `nufft_adj_2d` for each item in the batch, using threads.
    Returns a stacked output of shape [B, ...].
    
    Args:
      kspace_data_batched: batched data, shape [B, ...], each slice is for one sample
      kspace_traj_batched: shape [B, 2, ...] or [B, ...], each slice for one sample
      image_size: (H, W)
      norm_factor: normalization factor
      max_workers: how many worker threads to run in parallel
    """

    # kspace_data_batched = einx.rearrange("ch z len -> z ch len", kspace_data)
    
    ic(kspace_traj.shape)# 5, 48,2 length
    ic(kspace_data.shape)
    *batch_shape, _, length = kspace_traj.shape # * batch_shape = z, ch; _ = 2, length = length
    batch_size = np.prod(batch_shape, dtype=int)
    ic(batch_size) # 160
    outputs = [None] * batch_size
    kspace_traj_batched = einx.rearrange(
        "b... comp len -> (b...) comp len", kspace_traj
    )
    
    *channel_shape, length = kspace_data.shape[len(batch_shape) :] # kspace_data.shape[2 :]
    # kspace_data = einx.rearrange("ch z len -> z ch len", kspace_data)
    kspace_data_batched = kspace_data.contiguous().view(batch_size, *channel_shape, length)
    ic(kspace_data_batched.shape)
    
    def worker(i: int):
        # Each thread processes one slice:
        return nufft_adj_2d_Inner(
            kspace_data_batched[i], # ch sp*len
            kspace_traj_batched[i],#2 sp*len
            image_size,
            norm_factor,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(batch_size)]
        for i, f in enumerate(futures):
            outputs[i] = f.result()

    # Stack the results
    return einx.rearrange("(b...) ch... h w -> b... ch... h w", torch.stack(outputs, dim=0), b = batch_shape)

def radial_spokes_to_kspace_point(
    x: Shaped[KspaceSpokesData, "..."] | Shaped[KspaceSpokesTraj, "..."],
):
    return einx.rearrange(
        "... middle len -> ... (middle len)",
        x,
    )


def kspace_point_to_radial_spokes(
    x: Shaped[KspaceData, "..."],
    spoke_len: int,
):
    return einx.rearrange(
        "... (middle len) -> ... middle len",
        x,
        len=spoke_len,
    )


### test
# %%
# def data_binning_test():
#     data = torch.rand((15,80,2550,640))
#     spokes_per_contra = 75
#     phase_num = 5
#     spokes_per_phase = 15
#     contrast_num = 34
#     sorted_r_idx = torch.stack([ torch.randperm(spokes_per_contra) for i in range(contrast_num) ])
#     o = data_binning_jl(data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase)
#     return o.shape
# jl.data_binning(jl.Array(data.numpy()), jl.Array(sorted_r_idx.numpy()), contrast_num, spokes_per_contra, phase_num, spokes_per_phase)
# print(data_binning_test())
