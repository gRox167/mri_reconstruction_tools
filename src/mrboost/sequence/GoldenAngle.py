from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import einx
import numpy as np
import torch
from mrboost import computation as comp
from mrboost.bias_field_correction import n4_bias_field_correction_3d_complex
from mrboost.coil_sensitivity_estimation import get_csm_lowk_xyz
from mrboost.density_compensation.area_based_radial import (
    area_based_radial_density_compensation,
)
from mrboost.sequence.boilerplate import ReconArgs
from plum import dispatch
from mrboost.density_compensation_1 import ramp_density_compensation

@dataclass
class GoldenAngleArgs(ReconArgs):
    start_spokes_to_discard: int = field(default=0)  # to reach the steady state
    csm_lowk_hamming_ratio: Sequence[float] = field(default=(0.03, 0.03))
    density_compensation_func: Callable = field(
        default=area_based_radial_density_compensation
    )
    bias_field_correction: bool = field(default=False)
    return_csm: bool = field(default=True)
    return_multi_channel_image: bool = field(default=False)
    filtered_density_compensation_ratio: float = field(default=0.0)

    def __post_init__(self):
        super().__post_init__()


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: GoldenAngleArgs, z_dim_fft=True
):
    # if raw_data.device != torch.device(recon_args.device):
    #     raw_data = raw_data.to(recon_args.device)
    kspace_raw_data = raw_data * recon_args.amplitude_scale_factor
    kspace_traj = comp.generate_golden_angle_radial_spokes_kspace_trajectory(
        kspace_raw_data.shape[2], recon_args.spoke_len
    ) # input (spoke_num, spoke_len), output (2, spoke_num, spoke_len)
    if recon_args.partial_fourier_flag:
        kspace_data_centralized, kspace_data_mask = comp.centralize_kspace(
            kspace_data=kspace_raw_data,
            acquire_length=recon_args.partition_num,
            center_idx_in_acquire_lenth=recon_args.kspace_centre_partition_num,
            full_length=recon_args.slice_num,
            dim=1,
        )
    else:
        kspace_data_centralized = kspace_raw_data
        kspace_data_mask = None
    if z_dim_fft:
        # kspace_data_z = comp.batch_process(
        #     batch_size=1, device=recon_args.device
        # )(comp.ifft_1D)(kspace_data_centralized, dim=1, norm="backward")
        kspace_data_z = comp.ifft_1D(kspace_data_centralized, dim=1, norm="ortho")
        return dict(
            kspace_data_centralized=kspace_data_centralized,#raw data
            kspace_data_z=kspace_data_z,
            kspace_traj=kspace_traj,
            kspace_data_mask=kspace_data_mask,
        )
    else:
        return dict(
            kspace_data_centralized=kspace_data_centralized,
            kspace_traj=kspace_traj,
            kspace_data_mask=kspace_data_mask,
        )


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: GoldenAngleArgs,
    *args,
    **kwargs,
):
    kspace_data_centralized,kspace_data_z, kspace_traj = (
        data_preprocessed["kspace_data_centralized"][
            :, :, recon_args.start_spokes_to_discard :, :
        ],
        data_preprocessed["kspace_data_z"][
            :, :, recon_args.start_spokes_to_discard :, :
        ],
        data_preprocessed["kspace_traj"][:, recon_args.start_spokes_to_discard :, :],
    )
    
    if "csm" not in kwargs:
        csm = get_csm_lowk_xyz(
            kspace_data_centralized,
            kspace_traj,
            recon_args.im_size,
            recon_args.csm_lowk_hamming_ratio,
        )
    else:
        csm = kwargs["csm"]
        print("no csm calculated")
    kspace_density_compensation = recon_args.density_compensation_func(
        kspace_traj,
        im_size=recon_args.im_size,
        normalize=False,
        energy_match_radial_with_cartisian=True,
    )

    if recon_args.filtered_density_compensation_ratio > 0.0:
        length = kspace_density_compensation.shape[-1]
        front_idx = round((length * recon_args.filtered_density_compensation_ratio) / 2)
        back_idx = round(length * recon_args.filtered_density_compensation_ratio / 2)
        kspace_density_compensation[:, :front_idx] = kspace_density_compensation[
            :, front_idx : front_idx + 1
        ]
        kspace_density_compensation[:, -back_idx:] = kspace_density_compensation[
            :, -back_idx : -back_idx + 1
        ]


    kspace_data = comp.radial_spokes_to_kspace_point(
        kspace_data_z * kspace_density_compensation
    )
    kspace_traj = comp.radial_spokes_to_kspace_point(kspace_traj)

    img_multi_ch = comp.nufft_adj_2d(
        kspace_data,
        kspace_traj,
        recon_args.im_size,
        norm_factor=2 * np.sqrt(np.prod(recon_args.im_size)),
        # 2 because of readout_oversampling
    )
    img = einx.sum("[ch] slice w h", img_multi_ch * csm.conj())
    # img = einx.sum("[ch] slice w h", img_multi_ch * csm)

    if recon_args.bias_field_correction:
        img = n4_bias_field_correction_3d_complex(img)

    return_dict = {"image": img}
    if recon_args.return_csm:
        return_dict["csm"] = csm
    if recon_args.return_multi_channel_image:
        return_dict["image_multi_ch"] = img_multi_ch

    if len(return_dict) == 1:
        return return_dict["image"]
    else:
        return return_dict


@dispatch
def SoS_nufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: GoldenAngleArgs,
    *args,
    **kwargs,
):
    kspace_data_z, kspace_traj = (
        data_preprocessed["kspace_data_z"][
            :, :, recon_args.start_spokes_to_discard :, :
        ],
        data_preprocessed["kspace_traj"][:, recon_args.start_spokes_to_discard :, :],
    )

    kspace_density_compensation = recon_args.density_compensation_func(
        kspace_traj,
        im_size=recon_args.im_size,
        normalize=False,
        energy_match_radial_with_cartisian=True,
    )
    kspace_data = comp.radial_spokes_to_kspace_point(
        kspace_data_z * kspace_density_compensation
    )
    kspace_traj = comp.radial_spokes_to_kspace_point(kspace_traj)

    img_multi_ch = comp.nufft_adj_2d(
        kspace_data,
        kspace_traj,
        recon_args.im_size,
        norm_factor=2 * np.sqrt(np.prod(recon_args.im_size)),
        # 2 because of readout_oversampling
    )
    img = einx.sum("[ch] slice w h", img_multi_ch * img_multi_ch.conj())
    # img = einx.sum("[ch] slice w h", img_multi_ch * img_multi_ch.conj())

    return_dict = {"image": img}
    if recon_args.return_multi_channel_image:
        return_dict["image_multi_ch"] = img_multi_ch

    if len(return_dict) == 1:
        return return_dict["image"]
    else:
        return return_dict
