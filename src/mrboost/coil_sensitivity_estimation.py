import einx

# from keras import ops
import torch as ops

# import sigpy as sp
from . import computation as comp
from .density_compensation import (
    area_based_radial_density_compensation,
)


def get_csm_lowk_xy(
    kspace_data,
    kspace_traj,
    im_size,
    hamming_filter_ratio=0.05,
):
    ch, z, sp, spoke_len = kspace_data.shape

    kspace_density_compensation_ = area_based_radial_density_compensation(kspace_traj)

    spoke_len = kspace_data.shape[-1]
    W = comp.hamming_filter(nonzero_width_percent=hamming_filter_ratio, width=spoke_len)
    kspace_data = einx.multiply(
        "len, ch z sp len -> ch z sp len",
        W,
        kspace_data,
    )
    kspace_data = comp.ifft_1D(kspace_data * kspace_density_compensation_, dim=1)

    coil_sens = comp.nufft_adj_2d(
        comp.radial_spokes_to_kspace_point(kspace_data),
        comp.radial_spokes_to_kspace_point(kspace_traj),
        im_size,
    )

    img_sens_SOS = ops.sqrt(einx.sum("[ch] z h w", abs(coil_sens) ** 2))
    coil_sens = coil_sens / img_sens_SOS
    # coil_sens[ops.isnan(coil_sens)] = 0  # optional
    # coil_sens /= coil_sens.abs().max()
    return coil_sens


def get_csm_lowk_xyz(
    kspace_data,
    kspace_traj,
    im_size,
    hamming_filter_ratio=[0.05, 0.1],
):
    ch, z, sp, spoke_len = kspace_data.shape
    kspace_density_compensation_ = area_based_radial_density_compensation(kspace_traj)
    spoke_len = kspace_data.shape[-1]
    Wxy = comp.hamming_filter(
        nonzero_width_percent=hamming_filter_ratio[0], width=spoke_len
    )
    Wz = comp.hamming_filter(nonzero_width_percent=hamming_filter_ratio[1], width=z)

    kspace_data = einx.multiply(
        "len, kz, ch kz sp len -> ch kz sp len",
        Wxy,
        Wz,
        kspace_data,
    )
    kspace_data = comp.ifft_1D(kspace_data * kspace_density_compensation_, dim=1)

    coil_sens = comp.nufft_adj_2d(
        comp.radial_spokes_to_kspace_point(kspace_data),
        comp.radial_spokes_to_kspace_point(kspace_traj),
        im_size,
    )

    img_sens_SOS = ops.sqrt(einx.sum("[ch] z h w", ops.abs(coil_sens) ** 2))
    coil_sens = coil_sens / img_sens_SOS
    # coil_sens[ops.isnan(coil_sens)] = 0  # optional
    # coil_sens /= coil_sens.abs().max()

    return coil_sens
