# %%
from dataclasses import dataclass, field
from typing import Dict

import torch
from mrboost import computation as comp

# from mrboost.io_utils import *
from mrboost.sequence.GoldenAngle import GoldenAngleArgs

# from icecream import ic
from plum import dispatch


@dataclass
class DynGoldenAngleVibeArgs(GoldenAngleArgs):
    # duration_to_reconstruct: int = 1200  # in seconds
    # time_per_contrast: int = 60  # in seconds
    contra_num: int = 20
    spokes_per_contra: int = 320

    start_spokes_to_discard: int = field(init=False)  # to reach the steady state
    binning_spoke_slice: slice = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.start_spokes_to_discard = 10
        self.binning_spoke_slice = slice(0, self.contra_num * self.spokes_per_contra)
        self.return_multi_channel_image = True


@dispatch
def preprocess_raw_data(raw_data: torch.Tensor, recon_args: DynGoldenAngleVibeArgs):
    _dict = preprocess_raw_data.invoke(torch.Tensor, GoldenAngleArgs)(
        raw_data, recon_args
    )
    kspace_data_centralized = comp.data_binning_consecutive(
        _dict["kspace_data_centralized"][:, :, recon_args.start_spokes_to_discard :, :][
            :, :, recon_args.binning_spoke_slice
        ],
        recon_args.spokes_per_contra,
    )
    kspace_data_z = comp.data_binning_consecutive(
        _dict["kspace_data_z"][:, :, recon_args.start_spokes_to_discard :, :][
            :, :, recon_args.binning_spoke_slice
        ],
        recon_args.spokes_per_contra,
    )
    kspace_traj = comp.data_binning_consecutive(
        _dict["kspace_traj"][:, recon_args.start_spokes_to_discard :][
            :, recon_args.binning_spoke_slice
        ],
        recon_args.spokes_per_contra,
    )
    return dict(
        kspace_data_centralized=kspace_data_centralized,
        kspace_data_z=kspace_data_z,
        kspace_traj=kspace_traj,
    )


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: Dict[str, torch.Tensor],
    recon_args: DynGoldenAngleVibeArgs,
    return_multi_channel: bool = False,
    *args,
    **kwargs,
):
    images = []
    for t in range(recon_args.contra_num):
        # ic("reconstructing contrast:", t)
        # arguments = GoldenAngleArgs(
        #     recon_args.shape_dict,
        #     recon_args.mdh,
        #     recon_args.twixobj,
        #     recon_args.device,
        # )
        return_dict = mcnufft_reconstruct.invoke(
            Dict[str, torch.Tensor], GoldenAngleArgs
        )({k: v[t] for k, v in data_preprocessed.items()}, recon_args)

        images.append(return_dict["image_multi_ch"])
    return torch.stack(images, dim=0)
