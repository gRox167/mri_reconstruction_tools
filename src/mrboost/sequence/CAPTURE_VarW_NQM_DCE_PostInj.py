from dataclasses import dataclass, field
from typing import Callable, Dict

import einx
import numpy as np
import torch
from mrboost import computation as comp
from mrboost.coil_sensitivity_estimation import get_csm_lowk_xyz
from mrboost.density_compensation.area_based_radial import (
    area_based_radial_density_compensation,
)
from mrboost.sequence.GoldenAngle import GoldenAngleArgs
from plum import dispatch

# from icecream import ic


@dataclass
class CAPTURE_VarW_NQM_DCE_PostInj_Args(GoldenAngleArgs):
    # duration_to_reconstruct: int = 1200  # in seconds
    # time_per_contrast: int = 60  # in seconds
    phase_num: int = 5
    injection_time: int = 30
    duration_to_reconstruct: int = 340
    time_per_contrast: int = 10
    percentW: float = 12.5
    contra_num: int = field(init=False)
    start_spokes_to_discard: int = field(init=False)  # to reach the steady state
    binning_spoke_slice: slice = field(init=False)
    binning_start_idx: int = field(init=False)
    binning_end_idx: int = field(init=False)
    # which_slice: Union[int, Sequence] = -1,
    # which_contra: Union[int, Sequence] = -1,
    # which_phase: Union[int, Sequence] = -1,
    # cache_folder: Path = Path(".") / "cache",
    # device: torch.device = torch.device("cuda"),

    def __post_init__(self):
        super().__post_init__()

        self.total_partition_num = self.shape_dict["partition_num"]
        self.partition_num = (
            self.total_partition_num - 1
        )  # this number does not contain navigator

        self.start_spokes_to_discard = max(
            max(self.phase_num, 10),
            self.phase_num * np.ceil(10 / self.phase_num),
        )

        nSpokesToWorkWith = np.floor(self.duration_to_reconstruct / self.T)
        # per contrast from injection time. (Jan )
        # Exclude the first 10 prep spokes
        self.spokes_to_skip = round(
            (self.injection_time - self.time_per_contrast) / self.T
        )  # self.start_idx
        nSpokesPerContrast = np.floor(self.time_per_contrast / self.T)
        self.spokes_per_contra = int(
            nSpokesPerContrast - np.mod(nSpokesPerContrast, self.phase_num)
        )
        self.contra_num = int(np.floor(nSpokesToWorkWith / self.spokes_per_contra))
        self.spokes_per_phase = int(nSpokesPerContrast / self.phase_num)

        self.binning_start_idx = self.spokes_to_skip - self.start_spokes_to_discard
        self.binning_end_idx = (
            self.spokes_to_skip
            - self.start_spokes_to_discard
            + self.contra_num * self.spokes_per_contra
        )


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: CAPTURE_VarW_NQM_DCE_PostInj_Args
):
    nav = (
        einx.rearrange(
            "ch_num spoke_num spoke_len -> spoke_len spoke_num ch_num",
            raw_data[:, 0, recon_args.start_spokes_to_discard :, :],
        )
        * recon_args.amplitude_scale_factor
    )
    ch, rotation, respiratory_curve = comp.tuned_and_robust_estimation(
        navigator=nav.numpy(),
        percentW=recon_args.percentW,
        Fs=recon_args.Fs,
        FOV=recon_args.FOV,
        ndata=recon_args.spoke_len,
        device=recon_args.device,
    )
    # here rotation is index of 100 different degree, to get same with cihat, please+1
    respiratory_curve_contrast = einx.rearrange(
        "(contra spokes_per_contra) -> contra spokes_per_contra",
        respiratory_curve[recon_args.binning_start_idx : recon_args.binning_end_idx],
        contra=recon_args.contra_num,
        spokes_per_contra=recon_args.spokes_per_contra,
    )
    # separate the respiratory curve into different time periods (contrasts)

    sorted_r, sorted_r_idx = torch.sort(respiratory_curve_contrast, dim=-1)
    # in each of the contrast, we sort the respiratory curve in order to classify respiratory phases
    respiratory_curve = respiratory_curve[
        recon_args.binning_start_idx : recon_args.binning_end_idx
    ]

    kspace_raw_data = (
        raw_data[:, 1:, recon_args.start_spokes_to_discard :, :]
        * recon_args.amplitude_scale_factor
    )
    kspace_traj = comp.generate_golden_angle_radial_spokes_kspace_trajectory(
        raw_data.shape[2], recon_args.spoke_len
    )[:, recon_args.start_spokes_to_discard :]
    kspace_data_centralized, kspace_data_mask = comp.centralize_kspace(
        kspace_data=kspace_raw_data,
        acquire_length=recon_args.partition_num,
        center_idx_in_acquire_lenth=recon_args.kspace_centre_partition_num - 1,
        # -1 because of navigator, and this number is index started from 0
        full_length=recon_args.slice_num,
        dim=1,
    )
    # kspace_data_z = einx.vmap_with_axis(
    #     "ch [z] sp len -> ch [z] sp len", kspace_data_centralized,
    #     op = lambda x, axis: comp.fft_1D(x, dim=axis, norm="ortho")
    # )
    kspace_data_z = comp.ifft_1D(kspace_data_centralized, dim=1, norm="ortho")

    (
        _kspace_traj,
        _kspace_data_z,
        _kspace_data_centralized,
        _kspace_data_mask,
    ) = map(
        comp.data_binning,
        [
            kspace_traj[:, recon_args.binning_start_idx : recon_args.binning_end_idx],
            kspace_data_z[
                :, :, recon_args.binning_start_idx : recon_args.binning_end_idx
            ],
            kspace_data_centralized[
                :, :, recon_args.binning_start_idx : recon_args.binning_end_idx
            ],
            kspace_data_mask[
                :, :, recon_args.binning_start_idx : recon_args.binning_end_idx
            ],
        ],
        [sorted_r_idx] * 4,
        [recon_args.contra_num] * 4,
        [recon_args.spokes_per_contra] * 4,
        [recon_args.phase_num] * 4,
        [recon_args.spokes_per_phase] * 4,
    )

    return {
        "kspace_traj": _kspace_traj,
        "kspace_data_z": _kspace_data_z,
        "kspace_data_centralized": _kspace_data_centralized,
        "kspace_data_mask": _kspace_data_mask,
        "kspace_data_csm": kspace_data_centralized[
            :, :, recon_args.binning_start_idx : recon_args.binning_end_idx
        ],
        "kspace_traj_csm": kspace_traj[
            :, recon_args.binning_start_idx : recon_args.binning_end_idx
        ],
    }


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: CAPTURE_VarW_NQM_DCE_PostInj_Args,
    return_multi_channel: bool = False,
    density_compensation_func: Callable = area_based_radial_density_compensation,
    csm_xy_z_lowk_ratio=[0.05, 0.05],
    *args,
    **kwargs,
):
    kspace_data_centralized, kspace_data_z, kspace_traj = (
        data_preprocessed["kspace_data_centralized"],
        data_preprocessed["kspace_data_z"],
        data_preprocessed["kspace_traj"],
    )

    csm = get_csm_lowk_xyz(
        data_preprocessed["kspace_data_csm"],
        data_preprocessed["kspace_traj_csm"],
        recon_args.im_size,
        csm_xy_z_lowk_ratio,
    )
    images = []
    for t in range(recon_args.contra_num):
        phases = []
        for ph in range(recon_args.phase_num):
            print(f"reconstructing contrast {t + 1} phase {ph + 1}")
            _kspace_density_compensation = density_compensation_func(
                kspace_traj[t, ph],
                im_size=recon_args.im_size,
                normalize=False,
                energy_match_radial_with_cartisian=True,
            )
            _kspace_data = comp.radial_spokes_to_kspace_point(
                kspace_data_z[t, ph] * _kspace_density_compensation
            )
            _kspace_traj = comp.radial_spokes_to_kspace_point(kspace_traj[t, ph])
            img_multi_ch = comp.nufft_adj_2d(
                _kspace_data,
                _kspace_traj,
                recon_args.im_size,
                norm_factor=2 * np.sqrt(np.prod(recon_args.im_size)),
            )
            img = einx.sum("[ch] d w h", img_multi_ch * csm.conj())
            phases.append(img.cpu())
        images.append(torch.stack(phases, dim=0))
    return torch.stack(images, dim=0), csm
