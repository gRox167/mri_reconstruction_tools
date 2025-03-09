import einops as eo
import einx
import torch
import torchkbnufft as tkbn

from ..computation import (
    nufft_2d,
    nufft_adj_2d,
)


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
