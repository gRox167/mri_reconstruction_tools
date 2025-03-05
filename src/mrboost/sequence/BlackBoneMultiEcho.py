# %%
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import torch
from mrboost.density_compensation import (
    ramp_density_compensation,
)
from icecream import ic

# from mrboost.io_utils import *
from mrboost.sequence.GoldenAngle import (
    GoldenAngleArgs,
    SoS_nufft_reconstruct,
    mcnufft_reconstruct,
    preprocess_raw_data,
)

# from icecream import ic
from plum import dispatch


@dataclass
class BlackBoneMultiEchoArgs(GoldenAngleArgs):
    start_spokes_to_discard: int = field(default=5)  # to reach the steady state
    echo_num: int = field(default=3)
    csm_lowk_hamming_ratio: Sequence[float] = field(default=(0.05, 0.05))
    density_compensation_func: Callable = field(default=ramp_density_compensation)
    select_top_coils: int = field(default=0.95)
    bipolar_readout: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: BlackBoneMultiEchoArgs, *args, **kwargs
):
    assert recon_args.echo_num == raw_data.shape[0]
    echo, ch, kz, sp, len = raw_data.shape
    raw_data_ = raw_data.clone()
    data_list = []
    for e in range(recon_args.echo_num):
        if e % 2 == 1 and recon_args.bipolar_readout:
            # if readout mode is bipolar, even echo will have opposite readout direction to odd echo
            # map_twix will automatically flip the readout line with MDB flag of `REFLEX`
            # So we only need to correct the location of DC point.
            # other echo have DC point at index of 320, even echo have DC point at 319
            # so we need to roll 319 to 320
            raw_data_[e, :, :, :, :] = raw_data[e, :, :, :, :].roll(1, -1)
        data_dict = preprocess_raw_data.invoke(torch.Tensor, GoldenAngleArgs)(
            raw_data_[e],
            recon_args,
            z_dim_fft=True,
        )
        data_list.append(data_dict)
    return data_list


@dispatch
def mcnufft_reconstruct(
    data_preprocessed: List[Dict[str, torch.Tensor]],
    recon_args: BlackBoneMultiEchoArgs,
    *args,
    **kwargs,
):
    results = []
    for idx,e in enumerate(data_preprocessed):
        print(idx)
        result = mcnufft_reconstruct.invoke(Dict[str, torch.Tensor], GoldenAngleArgs)(e, recon_args, *args, **kwargs)
        ic(result.shape)
        results.append(result)
    return results
    # return [
    #     mcnufft_reconstruct.invoke(Dict[str, torch.Tensor], GoldenAngleArgs)(
    #         e,
    #         recon_args,
    #         *args,
    #         **kwargs,
    #     )
    #     for e in data_preprocessed
    # ]
    # return [
    #     mcnufft_reconstruct(e, recon_args, *args, **kwargs)
    #     for e in data_preprocessed
    # ]


@dispatch
def SoS_nufft_reconstruct(
    data_preprocessed: List[Dict[str, torch.Tensor]],
    recon_args: BlackBoneMultiEchoArgs,
    *args,
    **kwargs,
):
    return [
        SoS_nufft_reconstruct.invoke(Dict[str, torch.Tensor], GoldenAngleArgs)(
            e,
            recon_args,
            *args,
            **kwargs,
        )
        for e in data_preprocessed
    ]
