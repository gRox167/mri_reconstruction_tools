from typing import Sequence

import numpy as np
import torch
from jaxtyping import Shaped
from plum import dispatch, overload

from ..computation import (
    kspace_point_to_radial_spokes,
    nufft_2d,
    nufft_adj_2d,
    radial_spokes_to_kspace_point,
)
from ..type_utils import (
    KspaceSpokesTraj,
    KspaceTraj,
)


@overload
def ramp_density_compensation(
    kspace_traj: KspaceTraj,
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
):
    # _, l = kspace_traj.shape
    w = torch.norm(kspace_traj, dim=0)
    if normalize:
        #    Whether to normalize the density compensation.
        #    We normalize such that the energy of PSF = 1
        impulse = torch.zeros(
            (im_size[0], im_size[1]), dtype=torch.complex64, device=w.device
        )
        impulse[im_size[0] // 2, im_size[1] // 2] = 1
        wmax = (
            nufft_adj_2d(
                w * nufft_2d(impulse, kspace_traj, im_size),
                kspace_traj,
                im_size,
            )
            .abs()
            .max()
        )
        return w / wmax
    return w / w.abs().max()


@overload
def ramp_density_compensation(
    kspace_traj: KspaceSpokesTraj,
    im_size: Sequence[int] = (320, 320),
    normalize: bool = False,
    energy_match_radial_with_cartisian: bool = False,
):
    _, sp, len = kspace_traj.shape
    w = ramp_density_compensation(
        radial_spokes_to_kspace_point(kspace_traj), im_size, normalize
    )
    w = kspace_point_to_radial_spokes(w, len)
    if energy_match_radial_with_cartisian:
        nyquist_limit_spoke_num = round(np.pi * len // 2)
        w *= nyquist_limit_spoke_num / sp
    return w


@overload
def ramp_density_compensation(
    kspace_traj: Shaped[KspaceTraj, "b"],  # noqa: F821
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
):
    return torch.stack(
        [ramp_density_compensation(traj, im_size, normalize) for traj in kspace_traj]
    )


@overload
def ramp_density_compensation(
    kspace_traj: Shaped[KspaceSpokesTraj, "b"],  # noqa: F821
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
    energy_match_radial_with_cartisian: bool = False,
):
    return torch.stack(
        [
            ramp_density_compensation(
                traj, im_size, normalize, energy_match_radial_with_cartisian
            )
            for traj in kspace_traj
        ]
    )


@dispatch
def ramp_density_compensation(
    kspace_traj,
    im_size,
    *args,
    **wargs,
):
    pass
