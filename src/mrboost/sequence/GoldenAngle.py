from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import einx
import numpy as np
import torch
from mrboost import computation as comp
from mrboost.coil_sensitivity_estimation import (
    get_csm_lowk_xyz,
)
from mrboost.density_compensation import (
    ramp_density_compensation,
)
from mrboost.sequence.boilerplate import ReconArgs
from plum import dispatch


@dataclass
class GoldenAngleArgs(ReconArgs):
    adjnufft: Callable = field(init=False)
    nufft: Callable = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.adjnufft = lambda x, y: comp.nufft_adj_2d(x, y, self.im_size)
        self.nufft = lambda x, y: comp.nufft_2d(x, y, self.im_size)


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: GoldenAngleArgs, z_dim_fft=True
):
    kspace_raw_data = raw_data * recon_args.amplitude_scale_factor
    kspace_traj = comp.generate_golden_angle_radial_spokes_kspace_trajectory(
        kspace_raw_data.shape[2], recon_args.spoke_len
    )
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
            kspace_data_centralized=kspace_data_centralized,
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
    csm_lowk_hamming_ratio: Sequence[float] = [0.03, 0.03],
    density_compensation_func: Callable = ramp_density_compensation,
    discard_first_n_spokes: int = 5,
    *args,
    **kwargs,
):
    kspace_data_centralized, kspace_data_z, kspace_traj = (
        data_preprocessed["kspace_data_centralized"][:, :, discard_first_n_spokes:],
        data_preprocessed["kspace_data_z"][:, :, discard_first_n_spokes:],
        data_preprocessed["kspace_traj"][:, discard_first_n_spokes:],
    )

    csm = get_csm_lowk_xyz(
        kspace_data_centralized,
        kspace_traj,
        recon_args.im_size,
        csm_lowk_hamming_ratio,
    )

    kspace_density_compensation = density_compensation_func(
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
    img = einx.sum("[ch] slice w h", img_multi_ch * csm.conj())
    return img
