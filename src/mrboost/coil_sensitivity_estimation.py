from typing import Sequence

import einx
import numpy as np

# from keras import torch
import torch

# import sigpy as sp
# from dlboost.utils.tensor_utils import interpolate
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
        img_sens_SOS = torch.sqrt(einx.sum("[ch] z h w", abs(coil_sens) ** 2))
        csm = coil_sens / img_sens_SOS
    else:
        csm = coil_sens
    # coil_sens[torch.isnan(coil_sens)] = 0  # optional
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

    img_sens_SOS = torch.sqrt(einx.sum("[ch] z h w", torch.abs(coil_sens) ** 2))
    coil_sens = coil_sens / img_sens_SOS
    # coil_sens[torch.isnan(coil_sens)] = 0  # optional
    # coil_sens /= coil_sens.abs().max()

    return coil_sens


def get_csm_espirit(
    kspace_data,
    kspace_traj,
    image_size,
    kernel_size=(6, 6, 6),
    acs_size=(20, 20, 20),
    rank_threshold=0.02,
    crop_threshold=0.95,
    nsets=1,
    align_phase=False,
    kz_ifft_dim=1,
    normalize=True,
    esprit_device=torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu"),
    RO_oversampling_factor=2.0,
):
    """Estimate coil sensitivities with an ESPIRiT calibration pipeline. This only works for 3D radial sampling data."""
    if kspace_data.ndim != 4:
        raise ValueError(
            "Expected kspace_data with shape (ncoils, nz, nviews, nsamples)."
        )
    if kspace_traj is None:
        raise ValueError(
            "kspace_traj is required for ESPIRiT-based sensitivity estimation."
        )

    density_comp = area_based_radial_density_compensation(kspace_traj)
    kspace_data = einx.multiply(
        "ch kz sp len, sp len -> ch kz sp len", kspace_data, density_comp
    )
    nsamples = kspace_data.shape[-1]
    # extract calibration region
    if np.max(nsamples) >= 1024:
        cshape = list(np.asarray(image_size, dtype=int) // 16)
        scale = 1 / 16
    elif np.max(nsamples) >= 512:
        cshape = list(np.asarray(image_size, dtype=int) // 8)
        scale = 1 / 8
    elif np.max(nsamples) >= 256:
        cshape = list(np.asarray(image_size, dtype=int) // 4)
        scale = 1 / 4
    elif np.max(nsamples) >= 128:
        cshape = list(np.asarray(image_size, dtype=int) // 2)
        scale = 1 / 2
    else:
        cshape = [64] * 3
        scale = 1

    calib_data = _find_non_cartisian_acs(kspace_data, kspace_traj, cshape, scale=scale)
    print("calib_data.shape", calib_data.shape)
    original_device = calib_data.device
    maps_sets = _espirit_3d(
        calib_data.to(esprit_device),
        kernel_size=kernel_size,
        acs_size=acs_size,
        rank_threshold=rank_threshold,
        crop_threshold=crop_threshold,
        nsets=nsets,
        # image_size=image_size,
    ).to(original_device)
    # maps_sets = _espirit(
    #     calib_data, kernel_size[0], acs_size[0], rank_threshold, crop_threshold
    # )
    print("maps_sets.shape", maps_sets.shape)

    primary_maps = maps_sets[0]

    from dlboost.utils.tensor_utils import interpolate

    # primary_maps = interpolate(primary_maps, size=image_size)
    for _ in range(3):
        primary_maps = interpolate(
            primary_maps,
            scale_factor=2,
            mode="trilinear",
        )
    # primary_maps = resample(
    #     primary_maps, (primary_maps.shape[0],) + image_size, filt=True
    # )  # (ncoils, nz, ny, nx)

    if align_phase:
        #     phase_ref = einx.dot(
        #         "c z y x, c z y x -> z y x", coil_imgs, primary_maps.conj()
        #     )
        #     eps_phase = torch.finfo(primary_maps.real.dtype).eps
        #     phase = phase_ref / (torch.abs(phase_ref) + eps_phase)
        #     primary_maps = einx.multiply("c z y x, z y x -> c z y x", primary_maps, phase)
        # Phase relative to first map and crop
        primary_maps *= torch.conj(
            primary_maps[0] / (torch.abs(primary_maps[0]))  # + 1e-12)
        )

    if normalize:
        sos = torch.sqrt(einx.sum("[ch] z y x", torch.abs(primary_maps) ** 2))
        eps_norm = torch.finfo(primary_maps.real.dtype).eps
        primary_maps = primary_maps / (sos + eps_norm)

    return primary_maps


def _bounds(size, width):
    width = min(width, size)
    start = max((size - width) // 2, 0)
    return slice(start, start + width)


def _espirit_3d(
    kspace_data,
    kernel_size=(6, 6, 6),
    acs_size=(20, 20, 20),
    rank_threshold=0.02,
    crop_threshold=0.95,
    nsets=1,
    image_size=None,
    # device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    ncoils, depth, height, width = kspace_data.shape
    kd, kh, kw = kernel_size[0], kernel_size[1], kernel_size[2]
    # Extract calibration region.
    ACS = kspace_data[
        :,
        _bounds(depth, acs_size[0]),
        _bounds(height, acs_size[1]),
        _bounds(width, acs_size[2]),
    ]

    # Construct Hankel matrix. Use unfold to extract sliding windows
    # Unfold operates on one dimension at a time, so apply sequentially
    patches = (
        ACS.unfold(1, kd, 1).unfold(2, kh, 1).unfold(3, kw, 1)
    )  # (ncoils, rd, rh, rw, kd, kh, kw)
    A = einx.rearrange("c rd rh rw kd kh kw -> (rd rh rw) (c kd kh kw)", patches)

    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    r = torch.sum(S >= rank_threshold * S[0]).item()
    kernels = (
        torch.zeros(r, ncoils, depth, height, width, dtype=Vh.dtype, device=Vh.device)
        if image_size is None
        else torch.zeros(
            r,
            ncoils,
            image_size[0],
            image_size[1],
            image_size[2],
            dtype=Vh.dtype,
            device=Vh.device,
        )
    )
    kernels[:, :, _bounds(kd, kd), _bounds(kh, kh), _bounds(kw, kw)] = (
        Vh[:r].view(r, ncoils, kd, kh, kw).conj()
    )

    # scale = (depth * height * width) ** 0.5
    scale = (
        np.prod(image_size) ** 0.5
        if image_size is not None
        else (depth * height * width) ** 0.5
    )
    kernel_volume = (kd * kh * kw) ** 0.5
    kerimgs = comp.fft_nD(kernels).conj() * scale / kernel_volume  # why conj here?

    K = einx.rearrange("r c d h w -> (d h w) c r", kerimgs)
    # batch processed eigen-decomposition
    U = torch.zeros(
        nsets,
        ncoils,
        (depth * height * width) if image_size is None else np.prod(image_size),
        dtype=K.dtype,
        device=K.device,
    )
    for i, K_batched in enumerate(torch.split(K, 1024, dim=0)):
        # print("K_batched.shape", K_batched.shape)
        G_batched = torch.bmm(K_batched, K_batched.adjoint())
        # print("G_batched.shape", G_batched.shape)
        # perform eigen-decomposition
        w_batched, V_batched = torch.linalg.eig(G_batched)
        w_batched = w_batched.real
        sort_indices_batched = torch.argsort(w_batched, descending=True)
        mask = w_batched >= crop_threshold
        selected_indices_batched = sort_indices_batched[:, :nsets]
        U_batched = einx.get_at(
            "b c1 [c2], b i -> i c1 b",
            V_batched * mask[:, None, :],
            selected_indices_batched,
        )
        U[
            :,
            :,
            i * 1024 : (i + 1) * 1024 if (i + 1) * 1024 <= K.shape[0] else K.shape[0],
        ] = U_batched
    return (
        U.view(nsets, ncoils, depth, height, width)
        if image_size is None
        else U.view(nsets, ncoils, *image_size)
    )


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
        comp.fft_nD(kernels.flip(0).flip(1).flip(2).conj(), axes)
        * (sx * sy * sz) ** 0.5
        / (k**p) ** 0.5
    )

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    u, s, vh = torch.linalg.svd(
        kerimgs.view(sx, sy, sz, nc, n).reshape(-1, nc, n), full_matrices=True
    )
    print("s:", (s[:, :1] ** 2).min(), (s[:, :1] ** 2).max(), (s[:, :1] ** 2).mean())
    mask = s**2 > c

    # mask u (nvoxels, neigen, neigen)
    u = mask[:, None, :] * u

    # Reshape back to the original shape and assign to maps
    maps = u.view(sx, sy, sz, nc, nc)

    # transposex
    maps = maps.permute(4, 3, 2, 1, 0)

    return maps


def _find_non_cartisian_acs(
    kspace_data: torch.Tensor,
    ktraj: torch.Tensor,
    cal_shape: Sequence[int],
    scale=1 / 8,
):
    """Find approximate ACS region in non-Cartesian k-space data."""
    ch, kz, sp, sl = kspace_data.shape
    acs_data = kspace_data[
        :, _bounds(kz, cal_shape[0]), :, _bounds(sl, round(sl * scale))
    ]
    print("acs_data.shape", acs_data.shape)
    acs_data = einx.rearrange(
        "ch kz sp len -> ch kz (sp len)",
        acs_data,  # [..., RO_selection]
    )
    acs_traj = ktraj[:, :, _bounds(sl, round(sl * scale))].reshape(2, -1)
    # print(scale)
    # RO_selection = slice(
    #     sl // 2 - round(sl // 2 * scale), sl // 2 + round(sl // 2 * scale)
    # )
    # print("RO_selection", RO_selection)
    # dist_threshold = max(cal_shape[1], cal_shape[2]) / sl * torch.pi
    # r = torch.norm(ktraj, dim=0)
    # mask = r <= dist_threshold
    # num_acs_points = int(torch.sum(mask).item())
    # if num_acs_points < 10:
    #     raise ValueError(
    #         f"ACS region too small ({num_acs_points} points). Try increasing calib_shape."
    #     )
    # acs_data = torch.masked_select(acs_data, mask=mask)
    # apodize in RO direction and kz direction
    # acs_data = acs_data[:, :, :, RO_selection]
    # Wxy = comp.hanning_filter(nonzero_width_percent=1, width=acs_data.shape[-1])
    # Wz = comp.hanning_filter(nonzero_width_percent=1, width=acs_data.shape[1])
    # print("applied hanning window for ACS data")
    # acs_data = einx.multiply(
    #     "ch kz sp len, len, kz -> ch kz (sp len)",
    #     acs_data,
    #     Wxy,
    #     Wz,
    # )  # .view(ch, cal_shape[0], num_acs_points)
    # acs_traj = torch.masked_select(ktraj, mask=mask).view(2, num_acs_points)
    # acs_traj = ktraj[:, :, RO_selection].reshape(2, -1)

    cal_image = comp.nufft_adj_2d(
        acs_data,
        acs_traj / scale,
        (int(cal_shape[1]), int(cal_shape[2])),
    )
    cal_kspace_data = comp.fft_2D(cal_image, norm="ortho")
    return cal_kspace_data


def _expand_shapes(*shapes):
    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)

    shapes_exp = [np.asarray([1] * (max_ndim - len(shape)) + shape) for shape in shapes]
    shapes_exp = np.stack(shapes_exp, axis=0)  # (nshapes, max_ndim)
    shapes_exp = np.max(shapes_exp, axis=0)

    # restore original shape in non-padded portions
    shapes_exp = [list(shapes_exp[: -len(shape)]) + shape for shape in shapes]

    return tuple(shapes_exp)


def resize(input, oshape):
    """
    Resize with zero-padding or cropping.

    Adapted from SigPy [1].

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input tensor of shape ``(..., ishape)``.
    oshape : Iterable
        Output shape.

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Zero-padded or cropped tensor of shape ``(..., oshape)``.

    Examples
    --------
    >>> import torch
    >>> import deepmr

    We can pad tensors to desired shape:

    >>> x = torch.tensor([0, 1, 0])
    >>> y = deepmr.resize(x, [5])
    >>> y
    tensor([0, 0, 1, 0, 0])

    Batch dimensions are automatically expanded (pad will be applied starting from rightmost dimension):

    >>> x = torch.tensor([0, 1, 0])[None, ...]
    >>> x.shape
    torch.Size([1, 3])
    >>> y = deepmr.resize(x, [5]) # len(oshape) == 1
    >>> y.shape
    torch.Size([1, 5])

    Similarly, if oshape is smaller than ishape, the tensor will be cropped:

    >>> x = torch.tensor([0, 0, 1, 0, 0])
    >>> y = deepmr.resize(x, [3])
    >>> y
    tensor([0, 1, 0])

    Again, batch dimensions are automatically expanded:

    >>> x = torch.tensor([0, 0, 1, 0, 0])[None, ...]
    >>> x.shape
    torch.Size([1, 5])
    >>> y = deepmr.resize(x, [3]) # len(oshape) == 1
    >>> y.shape
    torch.Size([1, 3])

    References
    ----------
    [1] https://github.com/mikgroup/sigpy/blob/main/sigpy/util.py

    """

    if isinstance(oshape, int):
        oshape = [oshape]

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input

    # shift not supported for now
    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [
        min(i - si, o - so) for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)
    ]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape1, dtype=input.dtype, device=input.device)
    print("input.shape", input.shape)
    print("ishape1", ishape1)
    print("oshape1", oshape1)
    print("islice", islice)
    print("oslice", oslice)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output


def resample(input, oshape, filt=True, polysmooth=False):
    """
    Resample a n-dimensional signal.

    Parameters
    ----------
    input : np.ndarray | torch.Tensor
        Input tensor of shape ``(..., ishape)``.
    oshape : Iterable
        Output shape.
    filt : bool, optional
        If True and signal is upsampled (i.e., ``any(oshape > ishape)``),
        apply Fermi filter to limit ringing.
        The default is True.
    polysmooth : bool, optional
        If true, perform polynomial smoothing.
        The default is False. !!! NOT IMPLEMENTED YET !!!

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Resampled tensor of shape ``(..., oshape)``.

    """
    if isinstance(oshape, int):
        oshape = [oshape]

    # first, get number of dimensions
    ndim = len(oshape)
    axes = list(range(-ndim, 0))
    isreal = torch.isreal(input).all()

    # take fourier transform along last ndim axes
    freq = comp.fft_nD(input, dim=axes)

    # get initial and final shapes
    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    # build filter
    if filt and np.any(np.asarray(oshape1) > np.asarray(ishape1)):
        size = np.max(oshape1)
        width = np.min(oshape1)
        filt = fermi(ndim, size, width)
        filt = resize(filt, oshape1)  # crop to match dimension
    else:
        filt = None

    # resize in frequency space
    freq = resize(freq, oshape1)
    # if required, apply filtering
    if filt is not None:
        freq *= filt.to(freq.device)

    # transform back
    output = comp.ifft_nD(freq, axes)

    # take magnitude if original signal was real
    if isreal:
        output = abs(output)

    return output


def fermi(ndim, size, width=None):
    """
    Build a n-dimensional Fermi filter.

    This routine can handle isotropic ND matrices. User
    can specify the size of window support and the FWHM.
    The filter can be used in the context of image processing
    to mitigate ringing artifact [1].

    Parameters
    ----------
    ndim : int
        Number of dimensions (e.g., ``1=1D``, ``2=2D``, ``3=3D``).
    size : int
        Support of the window. Filter size will be ``ndim * [size]``.
    width : int, optional
        Full width half maximum of the filter.
        If ``None``, it is automatically set to ``size``. The default is ``None``.

    Returns
    -------
    filt : torch.Tensor
        Fermi window of shape ``ndim * [size]`` and ``FWHM = width``.

    Example
    -------
    >>> import deepmr

    We can design e.g., 1D, 2D or 3D filters as:

    >>> filt1d = deepmr.fermi(1, 128)
    >>> filt1d.shape
    torch.Size([128])
    >>> filt2d = deepmr.fermi(2, 128)
    torch.Size([128, 128])
    >>> filt3d = deepmr.fermi(3, 128)
    torch.Size([128, 128])

    Bu default, FWHM is equal to the support size:

    >>> (filt1d >= 0.5).sum()
    tensor(128)

    User can specify a smaller FWHM via ``width`` parameter:

    >>> filt1d = deepmr.fermi(1, 128, width=32)
    >>> filt1d.shape
    torch.Size([128])
    >>> (filt1d >= 0.5).sum()
    tensor(47)

    The discrepancy between nominal and actual FWHM is due to signal
    discretization.

    References
    ----------
    [1] Bernstein, M.A., Fain, S.B. and Riederer, S.J. (2001),
    Effect of windowing and zero-filled reconstruction of MRI data
    on spatial resolution and acquisition strategy.
    J. Magn. Reson. Imaging, 14: 270-280.
    https://doi.org/10.1002/jmri.1183

    """
    # default width
    if width is None:
        width = size

    # get radius
    radius = int(width // 2)

    # build grid, normalized so that u = 1 corresponds to window FWHM
    R = [
        torch.arange(int(-size // 2), int(size // 2), dtype=torch.float32)
        for n in range(ndim)
    ]

    # get transition width
    T = 20 * size  # width / 128

    # build center-out grid
    R = torch.meshgrid(*R, indexing="xy")
    R = torch.stack(R, dim=0)
    R = (R**2).sum(dim=0) ** 0.5

    # build filter
    filt = 1 / (1 + torch.exp((R - radius)) / T)
    filt /= filt.max()

    return filt
