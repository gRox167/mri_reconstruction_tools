from dataclasses import dataclass, field
from typing import Dict

import torch
from plum import dispatch

from mrboost.sequence.GoldenAngle import (
    GoldenAngleArgs,
    mcnufft_reconstruct,
    preprocess_raw_data,
)


@dataclass
class BlackBoneStationaryArgs(GoldenAngleArgs):
    start_spokes_to_discard: int = field(default=0)  # to reach the steady state

    def __post_init__(self):
        super().__post_init__()
        self.return_csm = True


@dispatch
def preprocess_raw_data(raw_data: torch.Tensor, recon_args: BlackBoneStationaryArgs):
    return preprocess_raw_data.invoke(torch.Tensor, GoldenAngleArgs)(
        raw_data, recon_args, z_dim_fft=True
    )


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: BlackBoneStationaryArgs,
    *args,
    **kwargs,
):
    return_dict = mcnufft_reconstruct.invoke(Dict[str, torch.Tensor], GoldenAngleArgs)(
        data_preprocessed, recon_args, *args, **kwargs
    )
    return return_dict["image"], return_dict["csm"]
    # kspace_data_centralized, kspace_traj = (
    #     data_preprocessed["kspace_data_centralized"],
    #     data_preprocessed["kspace_traj"],
    # )
    # csm = get_csm_lowk_xyz(
    #     kspace_data_centralized,
    #     kspace_traj,
    #     recon_args.im_size,
    #     recon_args.csm_lowk_hamming_ratio,
    #     # recon_args.device,
    # )
    # kspace_density_compensation = recon_args.density_compensation_func(
    #     kspace_traj,
    #     im_size=recon_args.im_size,
    #     normalize=False,
    #     energy_match_radial_with_cartisian=True,
    # )

    # kspace_data_centralized, kspace_traj, kspace_density_compensation = map(
    #     comp.radial_spokes_to_kspace_point,
    #     [kspace_data_centralized, kspace_traj, kspace_density_compensation],
    # )

    # kspace_data_z = comp.ifft_1D(kspace_data_centralized, dim=1, norm="ortho")
    # img_multi_ch = comp.nufft_adj_2d(
    #     kspace_data_z * kspace_density_compensation,
    #     kspace_traj,
    #     recon_args.im_size,
    #     norm_factor=2 * np.sqrt(np.prod(recon_args.im_size)),
    #     # 2 because of readout_oversampling
    # )

    # img = einx.sum("[ch] slice w h", img_multi_ch * csm.conj())
    # return img, csm
