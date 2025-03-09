import einops as eo
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import jit
from jax.tree_util import tree_map
from scipy.spatial import Voronoi

from .. import computation as comp
from ..torch_utils import as_real, jax_to_torch, torch_to_jax


@jit
def augment(kspace_traj_unique):
    r = 0.5  # spoke radius in kspace, should be 0.5
    kspace_traj_augmented = jnp.concatenate(
        (
            kspace_traj_unique,
            r * 1.005 * jnp.exp(1j * 2 * jnp.pi * jnp.arange(1, 257) / 256),
        )
    )
    kspace_traj_augmented = as_real(kspace_traj_augmented)
    return kspace_traj_augmented


def voronoi_density_compensation(
    kspace_traj: torch.Tensor,
    # im_size: Sequence[int],
    # grid_size: Optional[Sequence[int]] = None,
    device=torch.device("cpu"),
    *args,
    **kwargs,
):
    spoke_shape = eo.parse_shape(kspace_traj, "complx spokes_num spoke_len")
    kspace_traj = eo.rearrange(
        kspace_traj,
        "complx spokes_num spoke_len -> complx (spokes_num spoke_len)",
    )

    kspace_traj = torch.complex(kspace_traj[0], kspace_traj[1]).contiguous().to(device)
    with jax.default_device(jax.devices("cpu")[0]):
        kspace_traj = torch_to_jax(kspace_traj)
        kspace_traj = (
            kspace_traj / jnp.abs(kspace_traj).max() / 2
        )  # normalize to -0.5,0.5
        kspace_traj_unique, reverse_indices = jnp.unique(
            kspace_traj, return_inverse=True, axis=0
        )
        kspace_traj_len = kspace_traj_unique.shape[0]

        # draw a circle around spokes
        # plt.scatter(kspace_traj_augmented.real,kspace_traj_augmented.imag,s=0.5)
        kspace_traj_augmented = np.asarray(augment(kspace_traj_unique))
        vor = Voronoi(kspace_traj_augmented)

        def compute_area(region):
            if len(region) != 0:
                polygon = vor.vertices[region,]
                area = comp.polygon_area(polygon)
            else:
                area = float("inf")
            return area

        regions_area = jnp.array(
            tree_map(
                compute_area,
                vor.regions,
                is_leaf=lambda x: len(x) == 0 or isinstance(x[0], int),
            )
        )
        regions_area = regions_area[vor.point_region][:kspace_traj_len]
        regions_area /= jnp.sum(regions_area)
        regions_area = regions_area[reverse_indices]
        regions_area = eo.rearrange(
            regions_area,
            "(spokes_num spoke_len) -> spokes_num spoke_len",
            spokes_num=spoke_shape["spokes_num"],
        )
        regions_area = jax_to_torch(regions_area)
    regions_area[:, spoke_shape["spoke_len"] // 2] /= spoke_shape["spokes_num"]
    # Duplicate density for previously-removed points [i.e. DC points]
    return regions_area

    # fig = voronoi_plot_2d(vor,show_vertices=False,line_width=0.1,point_size=0.2)
    # plt.show()
