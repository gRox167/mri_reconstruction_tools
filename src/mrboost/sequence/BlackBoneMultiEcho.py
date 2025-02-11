# %%
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import einx
import torch
from mrboost.coil_selection import energy_based_coil_selection
from mrboost.density_compensation import (
    ramp_density_compensation,
)

# from mrboost.io_utils import *
from mrboost.sequence.GoldenAngle import (
    GoldenAngleArgs,
    mcnufft_reconstruct,
    preprocess_raw_data,
)

# from icecream import ic
from plum import dispatch


@dataclass
class BlackBoneMultiEchoArgs(GoldenAngleArgs):
    # start_spokes_to_discard: int = field(init=False)  # to reach the steady state
    nSubVolSlc: int = field(init=False)
    echo_num: int = field(default=3)
    csm_lowk_hamming_ratio: Sequence[float] = field(default=(0.05, 0.05))
    density_compensation_func: Callable = field(default=ramp_density_compensation)
    select_top_coils: int = field(default=0.95)

    def __post_init__(self):
        super().__post_init__()
        self.nSubVolSlc = round(self.slice_num * 3 / 4)


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: BlackBoneMultiEchoArgs, *args, **kwargs
):
    assert recon_args.echo_num == raw_data.shape[0]
    echo, ch, kz, sp, len = raw_data.shape
    print(recon_args.kspace_centre_partition_num)
    data_for_ch_selection = einx.rearrange(
        "echo ch kz sp len -> ch (echo kz sp len)",
        raw_data[
            :,
            :,
            :,
            # recon_args.kspace_centre_partition_num
            # - 5 : recon_args.kspace_centre_partition_num + 5,
            :,
            320 - 16 : 320 + 16,
        ],
    )
    selected_ch = energy_based_coil_selection(
        data_for_ch_selection,
        recon_args.select_top_coils,
    )
    data_list = []
    for e in range(recon_args.echo_num):
        data_list.append(
            preprocess_raw_data.invoke(torch.Tensor, GoldenAngleArgs)(
                raw_data[e, selected_ch, :, :, :], recon_args, z_dim_fft=True
            )
        )
    return data_list


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: List[Dict[str, torch.Tensor]],
    recon_args: BlackBoneMultiEchoArgs,
    csm_lowk_hamming_ratio: Sequence[float] = [0.03, 0.03],
    density_compensation_func: Callable = ramp_density_compensation,
    *args,
    **kwargs,
):
    return [
        mcnufft_reconstruct(
            e,
            recon_args,
            csm_lowk_hamming_ratio=recon_args.csm_lowk_hamming_ratio,
            density_compensation_func=recon_args.density_compensation_func,
            *args,
            **kwargs,
        )
        for e in data_preprocessed
    ]
