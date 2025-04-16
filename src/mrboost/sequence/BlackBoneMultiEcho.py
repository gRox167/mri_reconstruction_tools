# %%
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import einx
import torch
from mrboost.density_compensation.area_based_radial import (
    area_based_radial_density_compensation,
)
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
    # echo_num: int = field(default=3)
    csm_lowk_hamming_ratio: Sequence[float] = field(default=(0.03, 0.03))
    density_compensation_func: Callable = field(
        default=area_based_radial_density_compensation
    )
    bipolar_readout: bool = field(default=True)
    # select_top_coils: int = field(default=0.95)

    def __post_init__(self):
        super().__post_init__()


@dispatch
def preprocess_raw_data(
    raw_data: torch.Tensor, recon_args: BlackBoneMultiEchoArgs, *args, **kwargs
):
    # assert recon_args.echo_num == raw_data.shape[0]
    echo, ch, kz, sp, len = raw_data.shape
    # raw_data = raw_data[:, :, :, recon_args.start_spokes_to_discard :, :]
    raw_data_ = raw_data.clone()
    data_list = []
    for e in range(echo):
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
    # return [
    #     mcnufft_reconstruct.invoke(Dict[str, torch.Tensor], GoldenAngleArgs)(
    #         e,
    #         recon_args,
    #         *args,
    #         **kwargs,
    #     )
    #     for e in data_preprocessed
    # ]
    golden_angle_args = copy.copy(recon_args)
    golden_angle_args.return_csm = True

    return_data = mcnufft_reconstruct.invoke(Dict[str, torch.Tensor], GoldenAngleArgs)(
        data_preprocessed[0],
        golden_angle_args,
        *args,
        **kwargs,
    )
    csm = return_data["csm"]
    image_list = [return_data["image"]]

    for e in range(1, len(data_preprocessed)):
        return_data = mcnufft_reconstruct.invoke(
            Dict[str, torch.Tensor], GoldenAngleArgs
        )(
            data_preprocessed[e],
            golden_angle_args,
            csm=csm,
            *args,
            **kwargs,
        )
        image_list.append(return_data["image"])

    return image_list


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
