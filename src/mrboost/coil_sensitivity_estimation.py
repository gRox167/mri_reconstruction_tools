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
    kz_ifft_dim=1,
    normalize=True,
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
    if kz_ifft_dim is not None:
        kspace_data = comp.ifft_1D(
            kspace_data * kspace_density_compensation_, dim=kz_ifft_dim
        )
    else:
        kspace_data = kspace_data * kspace_density_compensation_
    coil_sens = comp.nufft_adj_2d(
        comp.radial_spokes_to_kspace_point(kspace_data),
        comp.radial_spokes_to_kspace_point(kspace_traj),
        im_size,
    )
    if normalize:
        img_sens_SOS = ops.sqrt(einx.sum("[ch] z h w", abs(coil_sens) ** 2))
        csm = coil_sens / img_sens_SOS
    else:
        csm = coil_sens
    # coil_sens[ops.isnan(coil_sens)] = 0  # optional
    # coil_sens /= coil_sens.abs().max()
    return csm


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


def get_csm_espirit():
    """Pytorch ESPIRIT implementation. Adapted for convenience from https://github.com/mikgroup/espirit-python/tree/master"""
    # TODO: implement this function with proper inputs and outputs as lowk method, you can refer to get_csm_lowk_xy
    # and for espirit implementation refer to espirit_cal function below
    pass


def espirit_cal(
    data, coord=None, dcf=None, shape=None, k=6, r=24, t=0.02, c=0.0, nsets=1
):
    """
    Derives the ESPIRiT [1] operator.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Multi channel k-space data.
    coord : np.ndarray | torch.Tensor, optional
        K-space trajectory of ``shape = (ncontrasts, nviews, nsamples, ndim)``.
        The default is ``None`` (Cartesian acquisition).
    dcf : np.ndarray | torch.Tensor, optional
        K-space density compensation of ``shape = (ncontrasts, nviews, nsamples)``.
        The default is ``None`` (no compensation).
    shape : Iterable[int] | optional
        Shape of the k-space after gridding. If not provided, estimate from
        input data (assumed on a Cartesian grid already).
        The default is ``None`` (Cartesian acquisition).
    k : int, optional
        k-space kernel size. The default is ``6``.
    r : int, optional
        Calibration region size. The default is ``24``.
    t : float, optional
        Rank of the auto-calibration matrix (A).
        The default is ``0.02``.
    c : float, optional
        Crop threshold that determines eigenvalues "=1".
        The defaults is ``0.95``.
    nsets : int, optional
        Number of set of maps to be returned.
        The default is ``1`` (conventional SENSE recon).

    Returns
    -------
    maps : np.ndarray | torch.Tensor
        Output coil sensitivity maps.

    Notes
    -----
    The input k-space ``data`` tensor is assumed to have the following shape:

    * **2Dcart:** ``(nslices, ncoils, ..., ny, nx)``.
    * **2Dnoncart:** ``(nslices, ncoils, ..., nviews, nsamples)``.
    * **3Dcart:** ``(nx, ncoils, ..., nz, ny)``.
    * **3Dnoncart:** ``(ncoils, ..., nviews, nsamples)``.

    For multi-contrast acquisitions, calibration is obtained by averaging over
    contrast dimensions.

    The output sensitivity maps are assumed to have the following shape:

    * **2Dcart:** ``(nsets, nslices, ncoils, ny, nx)``.
    * **2Dnoncart:** ``(nsets, nslices, ncoils, ny, nx)``.
    * **3Dcart:** ``(nsets, nx, ncoils, nz, ny)``.
    * **3Dnoncart:** ``(nsets, ncoils, nz, ny, nx)``.

    References
    ----------
    .. [1] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M.
           ESPIRiT--an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA.
           Magn Reson Med. 2014 Mar;71(3):990-1001. doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.

    """
    if isinstance(data, np.ndarray):
        isnumpy = True
    else:
        isnumpy = False

    while len(data.shape) < 5:
        data = data[None, ...]

    # keep shape
    if coord is not None:
        ndim = coord.shape[-1]
        if np.isscalar(shape):
            shape = ndim * [shape]
        else:
            shape = list(shape)[-ndim:]
        shape = [int(s) for s in shape]
    else:
        ndim = 2
        shape = list(data.shape[-2:])

    # extract calibration region
    if np.max(shape) >= 1024:
        cshape = list(np.asarray(shape, dtype=int) // 16)
    elif np.max(shape) >= 512:
        cshape = list(np.asarray(shape, dtype=int) // 8)
    elif np.max(shape) >= 256:
        cshape = list(np.asarray(shape, dtype=int) // 4)
    elif np.max(shape) >= 128:
        cshape = list(np.asarray(shape, dtype=int) // 2)
    else:
        cshape = [64] * ndim

    cal_data = _acs.find_acs(data, cshape, coord, dcf)

    # calculate maps
    maps = _espirit(cal_data.clone(), k, r, t, c)

    # select maps
    if nsets == 1:
        maps = maps[[0]]
    else:
        maps = maps[:nsets]

    # resample
    maps = _signal.resample(maps, shape)  # (nsets, ncoils, nz, ny, nx)

    # normalize
    maps_rss = _signal.rss(maps, axis=1, keepdim=True)
    maps = maps / maps_rss[[0]]

    # reformat
    if ndim == 2:  # Cartesian or 2D Non-Cartesian
        maps = maps.swapaxes(
            1, 2
        )  # (nsets, nslices, ncoils, ny, nx) / (nsets, nx, ncoils, nz, ny)

    # cast back to numpy if required
    if isnumpy:
        maps = maps.numpy(force=True)

    return maps, _signal.resize(cal_data, ndim * [r])


# %% local utils
def _espirit(X, k, r, t, c):
    # transpose
    X = X.permute(3, 2, 1, 0)

    # get shape
    sx, sy, sz, nc = X.shape

    sxt = (sx // 2 - r // 2, sx // 2 + r // 2) if (sx > 1) else (0, 1)
    syt = (sy // 2 - r // 2, sy // 2 + r // 2) if (sy > 1) else (0, 1)
    szt = (sz // 2 - r // 2, sz // 2 + r // 2) if (sz > 1) else (0, 1)

    # Extract calibration region.
    C = X[sxt[0] : sxt[1], syt[0] : syt[1], szt[0] : szt[1], :].to(
        dtype=torch.complex64
    )

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = torch.zeros(
        [(r - k + 1) ** p, k**p * nc], dtype=torch.complex64, device=X.device
    )

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
        for ydx in range(max(1, C.shape[1] - k + 1)):
            for zdx in range(max(1, C.shape[2] - k + 1)):
                block = C[xdx : xdx + k, ydx : ydx + k, zdx : zdx + k, :].to(
                    dtype=torch.complex64
                )
                A[idx, :] = block.flatten()
                idx += 1

    # Take the Singular Value Decomposition.
    U, S, VH = torch.linalg.svd(A, full_matrices=True)
    V = VH.conj().t()

    # Select kernels
    n = torch.sum(S >= t * S[0])
    V = V[:, :n]

    kxt = (sx // 2 - k // 2, sx // 2 + k // 2) if (sx > 1) else (0, 1)
    kyt = (sy // 2 - k // 2, sy // 2 + k // 2) if (sy > 1) else (0, 1)
    kzt = (sz // 2 - k // 2, sz // 2 + k // 2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = torch.zeros((sx, sy, sz, nc, n), dtype=torch.complex64, device=X.device)
    kerdims = [
        ((sx > 1) * k + (sx == 1) * 1),
        ((sy > 1) * k + (sy == 1) * 1),
        ((sz > 1) * k + (sz == 1) * 1),
        nc,
    ]
    for idx in range(n):
        kernels[kxt[0] : kxt[1], kyt[0] : kyt[1], kzt[0] : kzt[1], :, idx] = V[
            :, idx
        ].reshape(kerdims)

    # Take the iucfft
    axes = (0, 1, 2)
    kerimgs = (
        _fft.fft(kernels.flip(0).flip(1).flip(2).conj(), axes)
        * (sx * sy * sz) ** 0.5
        / (k**p) ** 0.5
    )

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    u, s, vh = torch.linalg.svd(
        kerimgs.view(sx, sy, sz, nc, n).reshape(-1, nc, n), full_matrices=True
    )
    mask = s**2 > c

    # mask u (nvoxels, neigen, neigen)
    u = mask[:, None, :] * u

    # Reshape back to the original shape and assign to maps
    maps = u.view(sx, sy, sz, nc, nc)

    # transpose
    maps = maps.permute(4, 3, 2, 1, 0)

    return maps


def find_acs(data, shape=None, coord=None, dcf=None):
    """
    Extract k-space calibration region.

    Parameters
    ----------
    data : np.ndarray | torch.Tensor
        Input k-space data.
    shape : int, optional
        Size of gridded k-space for Non-Cartesian imaging.
        The default is ``None``.
    coord : np.ndarray | torch.Tensor, optional
        K-space trajectory of ``shape = (ncontrasts, nviews, nsamples, ndim)``.
        The default is ``None`` (Cartesian acquisition).
    dcf : np.ndarray | torch.Tensor, optional
        K-space density compensation of ``shape = (ncontrasts, nviews, nsamples)``.
        The default is ``None`` (no compensation).

    Returns
    -------
    calib : np.ndarray | torch.Tensor
        Output calibration k-space data of shape ``(ncoils, nz, ny, nx).

    Notes
    -----
    The input k-space ``data`` tensor is assumed to have the following shape:

    * **2Dcart:** ``(nslices, ncoils, ncontrasts, ny, nx)``.
    * **2Dnoncart:** ``(nslices, ncoils, ncontrasts, nviews, nsamples)``.
    * **3Dcart:** ``(nx, ncoils, ncontrasts, nz, ny)``.
    * **3Dnoncart:** ``(ncoils, ncontrasts, nviews, nsamples)``.

    For multi-contrast acquisitions, calibration is obtained by averaging over
    contrast dimensions.

    """
    if coord is None:
        return _find_cart_acs(data)
    else:
        return _find_noncart_acs(data, shape, coord, dcf)


# %% local utils
def _find_cart_acs(kspace):
    """Adapted from https://github.com/mckib2/pygrappa/blob/main/pygrappa/find_acs.py"""

    # hardcoded
    coil_axis = 1

    # sum over contrasts
    kspace = kspace.sum(axis=-3)  # (nslices, ncoils, ny, nx)

    # convert to numpy
    if isinstance(kspace, torch.Tensor):
        istorch = True
        device = kspace.device
        kspace = kspace.numpy(force=True)
    else:
        istorch = False

    # cast to single precision
    kspace = kspace.astype(np.complex64)

    # move coil axis to the last dim
    kspace = np.moveaxis(kspace, coil_axis, -1)
    mask = np.abs(kspace[..., 0]) > 0

    # Start by finding the largest hypercube
    ctrs = [d // 2 for d in mask.shape]  # assume ACS is at center
    slices = [[c, c + 1] for c in ctrs]  # start with 1 voxel region
    while all(
        l > 0 and r < mask.shape[ii] for ii, (l, r) in enumerate(slices)
    ) and np.all(mask[tuple([slice(l - 1, r + 1) for l, r in slices])]):
        # expand isotropically until we can't no more
        slices = [[l0 - 1, r0 + 1] for l0, r0 in slices]

    # Stretch left/right in each dimension
    for dim in range(mask.ndim):
        # left: only check left condition on the current dimension
        while slices[dim][0] > 0 and np.all(
            mask[tuple([slice(l - (dim == k), r) for k, (l, r) in enumerate(slices)])]
        ):
            slices[dim][0] -= 1
        # right: only check right condition on the current dimension
        while slices[dim][1] < mask.shape[dim] and np.all(
            mask[tuple([slice(l, r + (dim == k)) for k, (l, r) in enumerate(slices)])]
        ):
            slices[dim][1] += 1

    # move back coil axis to original dim
    kspace = np.moveaxis(
        kspace[tuple([slice(l0, r0) for l0, r0 in slices] + [slice(None)])].copy(),
        -1,
        coil_axis,
    )

    # cast back to torch if required
    if istorch:
        kspace = torch.as_tensor(kspace, device=device, dtype=torch.complex64)

    kspace = kspace.swapaxes(0, 1)  # (ncoils, nz, ny, nx)
    kspace = _fft.fft(kspace, axes=(1,))

    return kspace


def _find_noncart_acs(kspace, cal_shape, coord, dcf):
    # get ndim
    ndim = coord.shape[-1]

    # collapse contrast dimension to view
    kspace = kspace.reshape(
        *kspace.shape[:2], 1, -1, kspace.shape[-1]
    )  # (nslices, ncoils, 1, ncontrasts * nviews, nsamples)
    coord = coord.reshape(
        1, -1, *coord.shape[-2:]
    )  # (1, ncontrasts * nviews, nsamples, ndim)
    if dcf is not None:
        dcf = dcf.reshape(1, -1, dcf.shape[-1])

    # find subset of coord along nsamples dim
    cabs = (coord**2).sum(axis=-1) ** 0.5  # sqrt(kx**2 + ky**2 + kz**2)
    ind = (
        cabs[0, 0] <= min(cal_shape) // 2
    )  # from (-mtx /2, mtx / 2) to (-cal / 2, cal / 2)

    # select kspace and coordinates
    cal_ksp = kspace[..., ind]
    cal_coord = coord[..., ind, :]

    if dcf is not None:
        cal_dcf = dcf[..., ind]
        cal_ksp = cal_dcf * cal_ksp

    # grid calibration
    cal_ksp = _fft.nufft_adj(
        cal_ksp, cal_coord, cal_shape, oversamp=2.0
    ).squeeze()  # (nslices, ncoils, ny, nx) or (ncoils, nz, ny, nx)
    if len(cal_ksp.shape) != 4:
        cal_ksp = cal_ksp[None, ...]
    if ndim == 2:
        cal_ksp = cal_ksp.swapaxes(0, 1)
    cal_ksp = _fft.fft(cal_ksp, axes=range(-3, 0))

    return torch.as_tensor(cal_ksp, dtype=torch.complex64)
