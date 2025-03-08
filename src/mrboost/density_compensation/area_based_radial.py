from typing import Sequence

import torch
from jaxtyping import Shaped
from plum import dispatch, overload

from ..computation import (
    kspace_point_to_radial_spokes,
    radial_spokes_to_kspace_point,
)
from ..type_utils import (
    KspaceSpokesTraj,
    KspaceTraj,
)


@overload
def area_based_radial_density_compensation(
    kspace_traj: KspaceSpokesTraj, *args, **kwargs
):
    """
    Analytical density compensation based on comparing radial and Cartesian sampling areas.

    Formula:
        w = π·r/(n·Δr) for r > 0
        w = π/(4n) for r = 0

    where r is radius in k-space, n is number of spokes, Δr is the radial step size.
    """
    _, n, length = kspace_traj.shape
    # Calculate radius (distance from k-space center)
    r = torch.norm(kspace_traj, dim=0)

    # Calculate delta_r (radial step size)
    delta_r = (
        2 * torch.pi / (length // 2)
    )  # Assuming normalized k-space with radius 0.5

    # Create weights using the analytical formula
    w = torch.zeros_like(r)

    # Handle r > 0 case: w = (π·r)/(n·Δr)
    nonzero_mask = r > 0
    w[nonzero_mask] = (torch.pi * r[nonzero_mask]) / (n * delta_r)

    # Handle r = 0 case: w = π/(4n)
    zero_mask = r == 0
    w[zero_mask] = torch.pi / (4 * n)

    return w


@overload
def area_based_radial_density_compensation(
    kspace_traj: KspaceTraj,
    spoke_len: int | None = None,
    spoke_num: int | None = None,  # noqa: F821
    *args,
    **kwargs,
) -> torch.Tensor:
    kj = kspace_point_to_radial_spokes(kspace_traj, spoke_num)
    w = area_based_radial_density_compensation(kj)
    return radial_spokes_to_kspace_point(w)


@overload
def area_based_radial_density_compensation(
    kspace_traj: Shaped[KspaceTraj, "b"],  # noqa: F821
    spoke_len: int | None = None,
    spoke_num: int | None = None,  # noqa: F821
    *args,
    **kwargs,
):
    return torch.stack(
        [
            area_based_radial_density_compensation(traj, spoke_len, spoke_num)
            for traj in kspace_traj
        ]
    )


@overload
def area_based_radial_density_compensation(
    kspace_traj: Shaped[KspaceSpokesTraj, "b"],  # noqa: F821
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
    energy_match_radial_with_cartisian: bool = False,
):
    return torch.stack(
        [
            area_based_radial_density_compensation(
                traj, im_size, normalize, energy_match_radial_with_cartisian
            )
            for traj in kspace_traj
        ]
    )


@dispatch
def area_based_radial_density_compensation(
    kspace_traj,
    im_size,
    *args,
    **kwargs,
):
    pass
