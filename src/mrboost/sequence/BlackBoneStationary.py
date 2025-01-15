# %%
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import einx
import numpy as np
import torch

# from icecream import ic
from plum import dispatch

from mrboost import computation as comp
from mrboost.coil_sensitivity_estimation import (
    get_csm_lowk_xyz,
)
from mrboost.density_compensation import (
    ramp_density_compensation,
)

# from mrboost.io_utils import *
from mrboost.sequence.GoldenAngle import GoldenAngleArgs


@dataclass
class BlackBoneStationaryArgs(GoldenAngleArgs):
    # start_spokes_to_discard: int = field(init=False)  # to reach the steady state
    nSubVolSlc: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.nSubVolSlc = round(self.slice_num * 3 / 4)


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: BlackBoneStationaryArgs
):
    return preprocess_raw_data.invoke(torch.Tensor, GoldenAngleArgs)(
        raw_data, recon_args, z_dim_fft=True
    )


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: BlackBoneStationaryArgs,
    csm_lowk_hamming_ratio: Sequence[float] = [0.03, 0.03],
    density_compensation_func: Callable = ramp_density_compensation,
    return_multi_channel: bool = False,
    *args,
    **kwargs,
):
    kspace_data_centralized, kspace_traj = (
        data_preprocessed["kspace_data_centralized"],
        data_preprocessed["kspace_traj"],
    )
    csm = get_csm_lowk_xyz(
        kspace_data_centralized,
        kspace_traj,
        recon_args.im_size,
        csm_lowk_hamming_ratio,
        # recon_args.device,
    )
    kspace_density_compensation = density_compensation_func(
        kspace_traj,
        im_size=recon_args.im_size,
        normalize=False,
        energy_match_radial_with_cartisian=True,
    )

    kspace_data_centralized, kspace_traj, kspace_density_compensation = map(
        comp.radial_spokes_to_kspace_point,
        [kspace_data_centralized, kspace_traj, kspace_density_compensation],
    )

    kspace_data_z = comp.ifft_1D(kspace_data_centralized, dim=1, norm="ortho")
    img_multi_ch = comp.nufft_adj_2d(
        kspace_data_z * kspace_density_compensation,
        kspace_traj,
        recon_args.im_size,
        norm_factor=2 * np.sqrt(np.prod(recon_args.im_size)),
        # 2 because of readout_oversampling
    )

    img = einx.sum("[ch] slice w h", img_multi_ch * csm.conj())
    return img, csm
