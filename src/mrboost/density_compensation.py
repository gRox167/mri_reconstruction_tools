from typing import Sequence

import einops as eo
import einx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchkbnufft as tkbn

# from icecream import ic
from jax import jit
from jax.tree_util import tree_map
from jaxtyping import Shaped
from plum import dispatch, overload
from scipy.spatial import Voronoi

# from toolz.functoolz import curry, pipe
from . import computation as comp
from .computation import (
    kspace_point_to_radial_spokes,
    nufft_2d,
    nufft_adj_2d,
    radial_spokes_to_kspace_point,
    nufft_2d_batched,
    nufft_adj_2d_batched,
    nufft_adj_2d_Inner,
)
from .torch_utils import as_real, jax_to_torch, torch_to_jax
from .type_utils import (
    KspaceSpokesTraj,
    KspaceTraj,
)
import concurrent.futures


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
    # device=torch.device("cpu"),
):
    spoke_shape = eo.parse_shape(kspace_traj, "complx spokes_num spoke_len")
    kspace_traj = eo.rearrange(
        kspace_traj,
        "complx spokes_num spoke_len -> complx (spokes_num spoke_len)",
    )

    # kspace_traj = torch.complex(kspace_traj[0], kspace_traj[1]).contiguous().to(device)
    kspace_traj = torch.complex(kspace_traj[0], kspace_traj[1]).contiguous()
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

def linear_expolate(weights):
    slope = weights[:,:,2]-weights[:,:,1]
    b = weights[:,:,1]-slope
    weights[:,:,0] = b
    slope1 = weights[:,:,-3]-weights[:,:,-2]
    b1 = weights[:,:,-2]-slope1
    weights[:,:,-1] = b1
    return weights

def pipe_density_compensation(kspace_traj, im_size, *args, **wargs):
    spoke_shape = eo.parse_shape(kspace_traj, "_ spokes_num spoke_len")
    w = tkbn.calc_density_compensation_function(
        ktraj=eo.rearrange(
            kspace_traj, "c spokes_num spoke_len -> c (spokes_num spoke_len)"
        ),
        im_size=im_size,
    )[0, 0]
    return eo.rearrange(
        w, " (spokes_num spoke_len) -> spokes_num spoke_len ", **spoke_shape
    )


def cihat_pipe_density_compensation(
    kspace_traj,
    nufft_ob=None,
    adjnufft_ob=None,
    im_size=(320, 320),
    device=torch.device("cpu"),
    *args,
    **wargs,
):
    prev_device = kspace_traj.device
    _, sp, l = kspace_traj.shape
    omega = einx.rearrange(
        "complx sp l -> complx (sp l)",
        kspace_traj,
    ).to(device)

    w = einx.rearrange(
        "l -> (sp l)",
        torch.linspace(-1, 1 - 2 / l, l, device=device).abs(),
        sp=sp,
        l=l,
    )
    impulse = torch.zeros(
        (im_size[0], im_size[1]), dtype=torch.complex64, device=device
    )
    impulse[im_size[0] // 2, im_size[1] // 2] = 1
    w = (
        w
        / nufft_adj_2d(w * nufft_2d(impulse, omega, im_size), omega, im_size)
        .abs()
        .max()
    )
    return einx.rearrange("(sp l) -> sp l", w, sp=sp, l=l).to(prev_device)



def ramp_density_compensation_Inner1(
    kspace_traj: KspaceTraj, # 2 length
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
        ) #create a [320,320] all 0 matrix
        impulse[im_size[0] // 2, im_size[1] // 2] = 1 # make the center as 1
        wmax = (
            nufft_adj_2d_Inner(
                w * nufft_2d(impulse, kspace_traj, im_size),
                kspace_traj,
                im_size,
            )
            .abs()
            .max()
        )
        return w / wmax
    return w / w.abs().max()


def ramp_density_compensation_Inner2(
    kspace_traj: KspaceSpokesTraj, # 2 sp, len
    im_size: Sequence[int] = (320, 320),
    normalize: bool = False,
    energy_match_radial_with_cartisian: bool = False,
):
    _, sp, len = kspace_traj.shape
    w = ramp_density_compensation_Inner1(
        radial_spokes_to_kspace_point(kspace_traj), im_size, normalize
    ) # get the initial density compensation weights for the transformed trajectory
    w = kspace_point_to_radial_spokes(w, len)
    if energy_match_radial_with_cartisian:
        nyquist_limit_spoke_num = round(np.pi * len // 2) # compute the number of spokes to satisfy the Nyquist limit for radial sampling
        w *= nyquist_limit_spoke_num / sp
    return w # dimension = [2, sp, len]

@overload
def ramp_density_compensation(
    kspace_traj: KspaceTraj, # 2. sp*len 
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
):
    w = ramp_density_compensation_Inner1(kspace_traj, im_size, normalize)
    return w
            
@overload
def ramp_density_compensation(
    kspace_traj: KspaceSpokesTraj, # 2. sp,len 
    im_size: Sequence[int] = (320, 320),
    normalize: bool = False,
    energy_match_radial_with_cartisian: bool = True,
):
    print('A')
    # kspace_traj = comp.radial_spokes_to_kspace_point(kspace_traj)
    w = ramp_density_compensation_Inner2(kspace_traj, im_size, normalize,energy_match_radial_with_cartisian)
    return w

@overload
def ramp_density_compensation(
    kspace_traj: Shaped[KspaceSpokesTraj, "b"], # b. 2. sp*len 
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
    energy_match_radial_with_cartisian: bool = False,
):
    return torch.stack(
        [
            ramp_density_compensation_Inner2(traj, im_size, normalize,energy_match_radial_with_cartisian)
            for traj in kspace_traj
        ]
    )


############## ty added ##########################
@overload
def ramp_density_compensation(
    kspace_traj: Shaped[KspaceSpokesTraj, "z ch"],#z ch, 2 sp ,len, 
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
    energy_match_radial_with_cartisian: bool = False,
):
    print("it is me")
    kspace_traj = einx.rearrange("b... complex sp len -> (b...) complex sp len", kspace_traj) # z*ch, c
    return torch.stack(
        [
            ramp_density_compensation_Inner2(
                traj, im_size, normalize, energy_match_radial_with_cartisian
            )
            for traj in kspace_traj # loop over all the trajectories
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
def ramp_density_compensation_A(
    kspace_traj: KspaceSpokesTraj,#2 sp, len, 
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
    energy_match_radial_with_cartisian: bool = False,
):
    # kspace_traj = einx.rearrange("b... complex sp len -> (b...) complex sp len", kspace_traj) # z*ch, c
    return torch.stack(
        [
            ramp_density_compensation_Inner2(
                traj, im_size, normalize, energy_match_radial_with_cartisian
            )
            for traj in kspace_traj # loop over all the trajectories
        ]
    )




def ramp_density_compensation_batched(
    kspace_traj: Shaped[KspaceSpokesTraj, "z"],#z ch, 2 sp ,len,
    im_size: Sequence[int] = (320, 320),
    normalize: bool = True,
    energy_match_radial_with_cartisian: bool = False,
    max_workers: int = 4,
):

    # kspace_traj = einx.rearrange("b... complex sp len -> (b...) complex sp len", kspace_traj)
    def worker(traj):
        return ramp_density_compensation_Inner2(    
            traj,
            im_size=im_size,
            normalize=normalize,
            energy_match_radial_with_cartisian=energy_match_radial_with_cartisian,
        )

    # 3) Use ThreadPoolExecutor to parallelize over the trajectory batch dimension
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, traj) for traj in kspace_traj]
        results = [f.result() for f in futures]

    # 4) Stack up the results, shape => [num_trajs, ...]
    return torch.stack(results, dim=0)
