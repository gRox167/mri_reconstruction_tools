import einx

# from keras import ops
import torch as ops

# import sigpy as sp
from . import computation as comp
from .density_compensation import (
    ramp_density_compensation,
)


def get_csm_lowk_xy(
    kspace_data,
    kspace_traj,
    im_size,
    hamming_filter_ratio=0.05,
):
    ch, z, sp, spoke_len = kspace_data.shape

    kspace_density_compensation_ = ramp_density_compensation(
        comp.radial_spokes_to_kspace_point(kspace_traj), im_size
    )
    kspace_density_compensation_ = comp.kspace_point_to_radial_spokes(
        kspace_density_compensation_, spoke_len
    )

    spoke_len = kspace_data.shape[-1]
    # bottom = ops.maximum(
    #     kspace_density_compensation_[:, spoke_len // 2],
    #     kspace_density_compensation_[:, spoke_len // 2 - 1],
    # )
    # kspace_density_compensation_[:, spoke_len // 2] = bottom
    # kspace_density_compensation_[:, spoke_len // 2 - 1] = bottom
    # print(kspace_density_compensation_[4, spoke_len // 2 - 1 : spoke_len // 2 + 1])
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
    kspace_density_compensation_ = ramp_density_compensation(
        kspace_traj, im_size, energy_match_radial_with_cartisian=True
    )
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


# def lowk_xy(
#     kspace_data,
#     kspace_traj,
#     adjnufft_ob,
#     hamming_filter_ratio=0.05,
#     batch_size=2,
#     device=torch.device("cpu"),
# ):
#     spoke_len = kspace_data.shape[-1]
#     W = comp.hamming_filter(nonzero_width_percent=hamming_filter_ratio, width=spoke_len)
#     spoke_lowpass_filter_xy = torch.from_numpy(W)

#     @comp.batch_process(batch_size=batch_size, device=device, batch_dim=0)
#     def apply_filter_and_nufft(kspace_data, filter, ktraj):
#         kspace_data = filter * kspace_data
#         kspace_data = comp.ifft_1D(kspace_data, dim=1)
#         # kspace_data = torch.flip(kspace_data, dims=(1,))
#         kspace_data = kspace_data / kspace_data.abs().max()
#         kspace_data = eo.rearrange(
#             kspace_data,
#             "ch_num slice_num spoke_num spoke_len -> ch_num slice_num (spoke_num spoke_len)",
#         )
#         # interp_mats = tkbn.calc_tensor_spmatrix(ktraj,im_size=adjnufft_ob.im_size.numpy(force=True))
#         img_dc = adjnufft_ob(kspace_data, ktraj)
#         # img_dc = eo.rearrange(img_dc, "slice_num ch_num h w -> ch_num slice_num h w")
#         # print(img_dc.shape)
#         return img_dc

#     coil_sens = apply_filter_and_nufft(
#         kspace_data,
#         filter=spoke_lowpass_filter_xy,
#         ktraj=eo.rearrange(
#             kspace_traj,
#             "complx spoke_num spoke_len -> complx (spoke_num spoke_len)",
#         ),
#     )

#     coil_sens = coil_sens[
#         :,
#         :,
#         # spoke_len // 2 - spoke_len // 4 : spoke_len // 2 + spoke_len // 4,
#         # spoke_len // 2 - spoke_len // 4 : spoke_len // 2 + spoke_len // 4,
#     ]
#     # coil_sens = torch.from_numpy(coil_sens)
#     img_sens_SOS = torch.sqrt(
#         eo.reduce(
#             coil_sens.abs() ** 2,
#             "ch_num slice_num height width -> () slice_num height width",
#             "mean",
#         )
#     )
#     coil_sens = coil_sens / img_sens_SOS
#     coil_sens[isnan(coil_sens)] = 0  # optional
#     # coil_sens /= coil_sens.abs().max()
#     return coil_sens


# def lowk_xyz(
#     kspace_data,
#     kspace_traj,
#     adjnufft_ob,
#     hamming_filter_ratio=[0.05, 0.1],
#     batch_size=2,
#     device=torch.device("cpu"),
#     **kwargs,
# ):
#     # "need to be used before kspace z axis ifft"
#     spoke_len = kspace_data.shape[-1]
#     slice_num = kspace_data.shape[1]
#     W = comp.hamming_filter(
#         nonzero_width_percent=hamming_filter_ratio[0], width=spoke_len
#     )
#     spoke_lowpass_filter_xy = torch.from_numpy(W)
#     Wz = comp.hamming_filter(
#         nonzero_width_percent=hamming_filter_ratio[1], width=slice_num
#     )
#     spoke_lowpass_filter_z = torch.from_numpy(Wz)

#     @comp.batch_process(batch_size=batch_size, device=device)
#     def apply_filter_and_nufft(kspace_data, filter_xy, filter_z, ktraj):
#         kspace_data = filter_xy * kspace_data
#         kspace_data = eo.einsum(filter_z, kspace_data, "b, a b c d -> a b c d")
#         kspace_data = comp.ifft_1D(kspace_data, dim=1)
#         # kspace_data = torch.flip(kspace_data, dims=(1,))
#         kspace_data = kspace_data / kspace_data.abs().max()
#         kspace_data = eo.rearrange(
#             kspace_data,
#             "ch_num slice_num spoke_num spoke_len -> ch_num slice_num (spoke_num spoke_len)",
#         ).contiguous()
#         img_dc = adjnufft_ob(kspace_data, ktraj)
#         # img_dc = eo.rearrange(img_dc, "ch_num slice_num h w -> ch_num slice_num h w")
#         return img_dc

#     coil_sens = apply_filter_and_nufft(
#         kspace_data,
#         filter_xy=spoke_lowpass_filter_xy,
#         filter_z=spoke_lowpass_filter_z,
#         ktraj=eo.rearrange(
#             kspace_traj,
#             "complx spoke_num spoke_len -> complx (spoke_num spoke_len)",
#         ),
#     )
#     img_sens_SOS = torch.sqrt(
#         eo.reduce(
#             coil_sens.abs() ** 2,
#             "ch_num slice_num height width -> () slice_num height width",
#             "mean",
#         )
#     )
#     coil_sens = coil_sens / img_sens_SOS
#     coil_sens[torch.isnan(coil_sens)] = 0  # optional
#     # coil_sens /= coil_sens.abs().max()
#     return coil_sens


# class CoilSensitivityEstimator:
#     def __init__(self, kspace_data, kspace_traj, batch_size, device) -> None:
#         self.kspace_data = kspace_data
#         self.kspace_traj = kspace_traj
#         self.batch_size = batch_size
#         self.device = device
#         self.coil_sens = field(default_factory=torch.Tensor)

#     def __getitem__(self, key):
#         return self.coil_sens[key]


# class Lowk_CSE(CoilSensitivityEstimator):
#     def __init__(
#         self,
#         kspace_data,
#         kspace_traj,
#         nufft_ob,
#         adjnufft_ob,
#         hamming_filter_ratio,
#         batch_size,
#         device,
#     ) -> None:
#         super().__init__(kspace_data, kspace_traj, batch_size, device)
#         self.adjnufft_ob = adjnufft_ob
#         self.nufft_ob = nufft_ob
#         self.hamming_filter_ratio = hamming_filter_ratio


# class Lowk_2D_CSE(Lowk_CSE):
#     def __init__(
#         self,
#         kspace_data,
#         kspace_traj,
#         nufft_ob,
#         adjnufft_ob,
#         im_size,
#         hamming_filter_ratio=0.05,
#         batch_size=2,
#         device=torch.device("cpu"),
#     ) -> None:
#         super().__init__(
#             kspace_data,
#             kspace_traj,
#             nufft_ob,
#             adjnufft_ob,
#             hamming_filter_ratio,
#             batch_size,
#             device,
#         )
#         kspace_density_compensation_ = cihat_pipe_density_compensation(
#             kspace_traj, nufft_ob, adjnufft_ob, im_size, device=self.device
#         )
#         self.coil_sens = lowk_xy(
#             kspace_data * kspace_density_compensation_,
#             kspace_traj,
#             adjnufft_ob,
#             hamming_filter_ratio,
#             batch_size=batch_size,
#             device=device,
#         )

#     def __getitem__(self, key):
#         current_contrast = key[0]
#         current_phase = key[1]
#         return super().__getitem__(key[2:])
#         # return self.coil_sens[key[2:]]


# class Lowk_3D_CSE(Lowk_CSE):
#     def __init__(
#         self,
#         kspace_data,
#         kspace_traj,
#         nufft_ob,
#         adjnufft_ob,
#         im_size,
#         hamming_filter_ratio=[0.05, 0.5],
#         batch_size=2,
#         device=torch.device("cpu"),
#     ) -> None:
#         super().__init__(
#             kspace_data,
#             kspace_traj,
#             nufft_ob,
#             adjnufft_ob,
#             hamming_filter_ratio,
#             batch_size,
#             device,
#         )
#         kspace_density_compensation_ = cihat_pipe_density_compensation(
#             kspace_traj, nufft_ob, adjnufft_ob, im_size, device=self.device
#         )
#         self.coil_sens = lowk_xyz(
#             kspace_data * kspace_density_compensation_,
#             kspace_traj,
#             adjnufft_ob,
#             hamming_filter_ratio,
#             batch_size=batch_size,
#             device=device,
#         )

#     def __getitem__(self, key):
#         return super().__getitem__(key[2:])


# class Lowk_5D_CSE(Lowk_CSE):
#     def __init__(
#         self,
#         kspace_data,
#         kspace_traj,
#         nufft_ob,
#         adjnufft_ob,
#         args,
#         hamming_filter_ratio=[0.05, 0.5],
#         batch_size=2,
#         device=torch.device("cpu"),
#     ) -> None:
#         super().__init__(
#             kspace_data,
#             kspace_traj,
#             nufft_ob,
#             adjnufft_ob,
#             hamming_filter_ratio,
#             batch_size,
#             device,
#         )
#         self.kspace_traj, self.kspace_data = map(
#             comp.data_binning,
#             [kspace_traj, kspace_data],
#             [args.sorted_r_idx] * 2,
#             [args.contra_num] * 2,
#             [args.spokes_per_contra] * 2,
#             [args.phase_num] * 2,
#             [args.spokes_per_phase] * 2,
#         )
#         # self.density_compensation_func = density_compensation_func

#     def __getitem__(self, key):
#         current_contrast = key[0]
#         current_phase = key[1]
#         kspace_traj = self.kspace_traj[current_contrast, current_phase]
#         kspace_density_compensation_ = cihat_pipe_density_compensation(
#             kspace_traj, self.nufft_ob, self.adjnufft_ob, device=self.device
#         )
#         return lowk_xyz(
#             self.kspace_data[current_contrast, current_phase]
#             * kspace_density_compensation_,
#             kspace_traj,
#             self.adjnufft_ob,
#             self.hamming_filter_ratio,
#             batch_size=self.batch_size,
#             device=self.device,
#         )


# class ESPIRIT(CoilSensitivityEstimator):
#     def __init__(self, kspace_data, kspace_traj, batch_size, device) -> None:
#         super().__init__(kspace_data, kspace_traj, batch_size, device)

#     def __getitem__(self, key):
#         return super().__getitem__(key)


# class Espirit_CSE(sp.app.App):
#     """ESPIRiT calibration.
#     Currently only supports outputting one set of maps.
#     Args:
#         ksp (array): k-space array of shape [num_coils, n_ndim, ..., n_1]
#         calib (tuple of ints): length-3 image shape. DHW
#         thresh (float): threshold for the calibration matrix.
#         kernel_width (int): kernel width for the calibration matrix.
#         max_power_iter (int): maximum number of power iterations.
#         device (Device): computing device.
#         crop (int): cropping threshold.
#     Returns:
#         array: ESPIRiT maps of the same shape as ksp.
#     References:
#         Martin Uecker, Peng Lai, Mark J. Murphy, Patrick Virtue, Michael Elad,
#         John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
#         ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
#         Where SENSE meets GRAPPA.
#         Magnetic Resonance in Medicine, 71:990-1001 (2014)
#     """
#     def __init__(self, kspace_data, kspace_traj, im_size, calib_width=(8,24,24),
#                  thresh=0.02, kernel_width=6, crop=0.95,
#                  max_iter=100, batch_size=2, device=sp.cpu_device,
#                  output_eigenvalue=False,show_pbar=True):
#         self.device = device
#         self.output_eigenvalue = output_eigenvalue
#         self.crop = crop
#         self.coil_sens = field(default_factory=torch.Tensor)

#         shape_dict = eo.parse_shape(kspace_data, 'ch_num slice_num spoke_num spoke_len ')
#         img_ndim = len(im_size)
#         num_coils = shape_dict['ch_num']

#         # kspace density compensation
#         kspace_density_compensation = pipe_density_compensation(kspace_traj, im_size)

#         # we need only the center of k-space, which we have fully samples during scanning
#         center_crop_shape = [ num_coils, 2*calib_width[0],shape_dict['spoke_num'],2*calib_width[2]]
#         kspace_data = center_crop(kspace_data*kspace_density_compensation,center_crop_shape)
#         # from matplotlib import pyplot as plt
#         # plt.imshow(torch.abs(kspace_data[0,5,250:300,:]))
#         # assert False, "breakpoint"
#         kspace_traj = center_crop(kspace_traj,[2,shape_dict['spoke_num'],calib_width[1]*2])
#         # from matplotlib import pyplot as plt
#         # plt.scatter(x=kspace_traj[0,250:300],y=kspace_traj[1,250:300])
#         # assert False, "breakpoint"
#         # print('test',kspace_data.shape)

#         # resample the golden-angle data to cartesian k-space to get calib region
#         # do we need kspace-density compensation?
#         kspace_data = eo.rearrange(
#             kspace_data, 'ch_num slice_num spoke_num spoke_len -> slice_num ch_num (spoke_num spoke_len)'
#             ).contiguous()
#         kspace_traj = eo.rearrange(kspace_traj, 'c spoke_num spoke_len -> c (spoke_num spoke_len)')
#         grid_size = (calib_width[1]*2,calib_width[2]*2)
#         kspace_interp_ob = tkbn.KbInterpAdjoint(im_size=grid_size,grid_size=grid_size,numpoints=6)#,n_shift=(grid_size[0],grid_size[0]))
#         kspace_data_cartesian = torch.fft.ifftshift(kspace_interp_ob(kspace_data,kspace_traj),[-1,-2])
#         kspace_data_cartesian = eo.rearrange(
#             kspace_data_cartesian, 'slice_num ch_num h w -> ch_num slice_num h w')

#         # print(kspace_data_cartesian)
#         from matplotlib import pyplot as plt
#         plt.imshow(torch.abs(kspace_data_cartesian[0,5]),vmin=0,vmax=0.5)
#         assert False, "breakpoint"
#         kspace_data = sp.from_pytorch(torch.view_as_real(kspace_data_cartesian), iscomplex=True)
#         print(kspace_data.shape)
#         # print(kspace_data[0,8])
#         with sp.get_device(kspace_data):
#             # Get calibration region
#             calib_shape = (num_coils,) + calib_width
#             print(calib_shape)
#             calib = sp.resize(kspace_data, calib_shape)
#             print(calib)
#             calib = sp.to_device(calib, device)
#         print(calib)

#         xp = self.device.xp
#         with self.device:
#             # Get calibration matrix.
#             # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
#             mat = sp.array_to_blocks(
#                 calib, [kernel_width] * img_ndim, [1] * img_ndim)
#             mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
#             mat = mat.transpose([1, 0, 2])
#             mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])
#             # print(mat)

#             # Perform SVD on calibration matrix
#             _, S, VH = xp.linalg.svd(mat, full_matrices=False)
#             VH = VH[S > thresh * S.max(), :]

#             # Get kernels
#             num_kernels = len(VH)
#             kernels = VH.reshape(
#                 [num_kernels, num_coils] + [kernel_width] * img_ndim)
#             # img_shape = kspace_data.shape[1:]
#             img_shape = (shape_dict['slice_num'],)+im_size

#             # Get covariance matrix in image domain
#             AHA = xp.zeros(img_shape[::-1] + (num_coils, num_coils),
#                            dtype=kspace_data.dtype)
#             for kernel in kernels:
#                 print(kernel.shape)
#                 img_kernel = sp.ifft(sp.resize(kernel, (num_coils,)+img_shape),
#                                      axes=range(-img_ndim, 0))
#                 aH = xp.expand_dims(img_kernel.T, axis=-1)
#                 a = xp.conj(aH.swapaxes(-1, -2))
#                 AHA += aH @ a

#             AHA *= (sp.prod(img_shape) / kernel_width**img_ndim)
#             self.mps = xp.ones(img_shape[::-1]+(num_coils,) + (1, ), dtype=kspace_data.dtype)

#             def forward(x):
#                 with sp.get_device(x):
#                     return AHA @ x

#             def normalize(x):
#                 with sp.get_device(x):
#                     return xp.sum(xp.abs(x)**2,
#                                   axis=-2, keepdims=True)**0.5

#             alg = sp.alg.PowerMethod(
#                 forward, self.mps, norm_func=normalize,
#                 max_iter=max_iter)
#         super().__init__(alg, show_pbar=show_pbar)

#     def _output(self):
#         xp = self.device.xp
#         with self.device:
#             # Normalize phase with respect to first channel
#             mps = self.mps.T[0]
#             mps *= xp.conj(mps[0] / xp.abs(mps[0]))

#             # Crop maps by thresholding eigenvalue
#             max_eig = self.alg.max_eig.T[0]
#             mps *= max_eig > self.crop

#         if self.output_eigenvalue:
#             return mps, max_eig
#         else:
#             return mps

#     def __getitem__(self, key):
#         current_contrast = key[0]
#         current_phase = key[1]
#         return self.coil_sens.__getitem__(key[2:])
