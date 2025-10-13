import os
import time
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Literal

import einx
import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import torch
import zarr
from einops import parse_shape
from jaxtyping import ArrayLike
from matplotlib.colors import Normalize
from plum import dispatch, overload
from torch import Tensor
from xarray import DataArray


def plot_mvf_with_image(
    img,
    mvf,
    location,
    stride,
    vmin=None,
    vmax=None,
    cnorm=Normalize(-10, 10),
    scale_units="xy",
    fig=None,
    axes=None,
):
    z0, y0, x0 = location  # Now unpacked as (z,y,x)
    if fig is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        animate_flag = True
    else:
        ax1, ax2, ax3 = axes
        animate_flag = False

    # 1. Axial plane (x-y plane, fixed z)
    ax1.imshow(img[z0], cmap="gray", vmin=vmin, vmax=vmax)
    y_ax, x_ax = np.mgrid[0 : mvf.shape[3] : stride, 0 : mvf.shape[4] : stride]
    z_ax = mvf[
        0, 0, z0, slice(None, None, stride), slice(None, None, stride)
    ]  # z component
    y_ax_vec = mvf[
        0, 1, z0, slice(None, None, stride), slice(None, None, stride)
    ]  # y component
    x_ax_vec = mvf[
        0, 2, z0, slice(None, None, stride), slice(None, None, stride)
    ]  # x component
    q_ax = ax1.quiver(
        x_ax,
        y_ax,
        -x_ax_vec,
        -y_ax_vec,
        z_ax,
        cmap="coolwarm",
        angles="xy",
        scale_units=scale_units,
        scale=1,
        norm=cnorm,
    )
    cbar = plt.colorbar(q_ax, ax=ax1, label="Z displacement") if animate_flag else None
    ax1.axhline(y0, color="r", linestyle="--", alpha=0.3)
    ax1.axvline(x0, color="r", linestyle="--", alpha=0.3)
    ax1.set_title(f"Axial plane (z={z0})\nColors: z-direction")

    # 2. Sagittal plane (y-z plane, fixed x)
    ax2.imshow(img[:, :, x0], cmap="gray", vmin=vmin, vmax=vmax)
    z_sag, y_sag = np.mgrid[0 : mvf.shape[2] : stride, 0 : mvf.shape[3] : stride]
    x_sag = mvf[
        0, 2, slice(None, None, stride), slice(None, None, stride), x0
    ]  # x component
    y_sag_vec = mvf[
        0, 1, slice(None, None, stride), slice(None, None, stride), x0
    ]  # y component
    z_sag_vec = mvf[
        0, 0, slice(None, None, stride), slice(None, None, stride), x0
    ]  # z component
    q_sag = ax2.quiver(
        y_sag,
        z_sag,
        -y_sag_vec,
        -z_sag_vec,
        x_sag,
        cmap="coolwarm",
        angles="xy",
        scale_units=scale_units,
        scale=1,
        norm=cnorm,
    )
    cbar = plt.colorbar(q_sag, ax=ax2, label="X displacement") if animate_flag else None
    ax2.axhline(z0, color="r", linestyle="--", alpha=0.5)
    ax2.axvline(y0, color="r", linestyle="--", alpha=0.5)
    ax2.set_title(f"Sagittal plane (x={x0})\nColors: x-direction")

    # 3. Coronal plane (x-z plane, fixed y)
    ax3.imshow(img[:, y0, :], cmap="gray", vmin=vmin, vmax=vmax)
    z_cor, x_cor = np.mgrid[0 : mvf.shape[2] : stride, 0 : mvf.shape[4] : stride]
    y_cor = mvf[
        0, 1, slice(None, None, stride), y0, slice(None, None, stride)
    ]  # y component
    x_cor_vec = mvf[
        0, 2, slice(None, None, stride), y0, slice(None, None, stride)
    ]  # x component
    z_cor_vec = mvf[
        0, 0, slice(None, None, stride), y0, slice(None, None, stride)
    ]  # z component
    q_cor = ax3.quiver(
        x_cor,
        z_cor,
        -x_cor_vec,
        -z_cor_vec,
        y_cor,
        cmap="coolwarm",
        angles="xy",
        scale_units=scale_units,
        scale=1,
        norm=cnorm,
    )
    cbar = plt.colorbar(q_cor, ax=ax3, label="Y displacement") if animate_flag else None
    ax3.axhline(z0, color="r", linestyle="--", alpha=0.5)
    ax3.axvline(x0, color="r", linestyle="--", alpha=0.5)
    ax3.set_title(f"Coronal plane (y={y0})\nColors: y-direction")

    # plt.tight_layout()
    plt.axis("off")

    return fig, ax1, ax2, ax3


def rotate_mvf_torch2med(_mvf):
    mvf = _mvf.clone()
    mvf[:, 1] = -mvf[:, 1]
    mvf = mvf.flip(3)
    # mvf = einx.rearrange(
    #     "b (vz+vy+vx) d h w -> b (vz+vx+vy) d w h", mvf.flip(2), vx=1, vy=1, vz=1
    # )
    return mvf


def rotate_img_torch2med(_img):
    return _img.flip(1)


def torch_to_nii_direction(data):
    if data.dim() == 3:
        # Assuming data is in the format (d, h, w)
        data = einx.rearrange("d h w -> w h d", data)
        return data
    elif data.dim() == 4:
        # Assuming data is in the format (d, h, w, c)
        data = einx.rearrange("d h w c -> w h d c", data)
        return data
    else:
        raise ValueError(
            "Input tensor must be 3D or 4D with shape (d, h, w) or (d, h, w, c)"
        )


@dispatch
def nii_to_torch_direction(data: Tensor):
    if data.dim() == 3:
        # Assuming data is in the format (d, h, w)
        return einx.rearrange("w h d -> d h w", data)

    elif data.dim() == 4:
        # Assuming data is in the format (d, h, w, c)
        return einx.rearrange("w h d c -> d h w c", data)
    else:
        raise ValueError(
            "Input tensor must be 3D or 4D with shape (w, h, d) or (w, h, d, c)"
        )


@dispatch
def nii_to_torch_direction(data: ArrayLike, flip=True):
    if len(data.shape) == 3:
        # Assuming data is in the format (d, h, w)
        return einx.rearrange("w h d -> d h w", data)
    elif len(data.shape) == 4:
        # Assuming data is in the format (d, h, w, c)
        return einx.rearrange("w h d c -> d h w c", data)
    else:
        raise ValueError(
            "Input tensor must be 3D or 4D with shape (w, h, d) or (w, h, d, c)"
        )


@dispatch
def plot_3D(
    image: Tensor,
    vmin=None,
    vmax=None,
    location=(0, 0, 0),
    scale=None,
    show_crosshairs=False,
    colorbar=None,
    title=None,
    axes=None,
    fig=None,
    figsize=(30, 10),
    tight_layout=True,
):
    z, y, x = location
    if fig is not None:
        if axes is not None:
            pass
        else:
            axes = fig.subplots(1, 3)
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    if scale is not None:
        # use nn.functional.interpolate to scale the image
        image = (
            torch.nn.functional.interpolate(
                image.unsqueeze(0).unsqueeze(0), scale_factor=scale, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )

    axes[0].imshow(image[z, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].title.set_text("Axial")
    axes[1].imshow(image[:, :, x], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].title.set_text("Sagittal")
    axes[2].imshow(image[:, y, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].title.set_text("Coronal")

    # Add crosshairs if requested
    if show_crosshairs:
        # Axial view (shows y-x plane)
        axes[0].axhline(y=y, color="r", linestyle="--", alpha=0.7)
        axes[0].axvline(x=x, color="r", linestyle="--", alpha=0.7)

        # Sagittal view (shows z-y plane)
        axes[1].axhline(y=z, color="r", linestyle="--", alpha=0.7)
        axes[1].axvline(x=y, color="r", linestyle="--", alpha=0.7)
        # Coronal view (shows z-x plane)
        axes[2].axhline(y=z, color="r", linestyle="--", alpha=0.7)
        axes[2].axvline(x=x, color="r", linestyle="--", alpha=0.7)

    for ax in axes:
        ax.axis("off")
    if title is not None:
        # set subfigure title
        fig.suptitle(title, fontsize=20)
    if tight_layout:
        plt.tight_layout()
    return fig, axes


@dispatch
def plot_3D(image: np.ndarray, *args, **kwargs):
    return plot_3D(torch.from_numpy(image), *args, **kwargs)


@dispatch
def plot_3D(
    image: nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image, channel=0, *args, **kwargs
):
    fdata = image.get_fdata()
    if fdata.ndim == 4:
        fdata = fdata[..., channel]
    else:
        fdata = fdata

    return plot_3D(nii_to_torch_direction(fdata).copy(), *args, **kwargs)


@dispatch
def plot_3D(image: str, *args, **kwargs):
    if image.endswith(".nii") or image.endswith(".nii.gz"):
        image = nib.load(image)
    elif image.endswith(".npy"):
        image = np.load(image)
    elif image.endswith(".zarr"):
        image = zarr.open(image, mode="r")
    return plot_3D(image, *args, **kwargs)


def get_raw_data(
    dat_file_location: Path,
    multi_echo=False,
    sampling_pattern: Literal["Cartesian", "radial"] = "radial",
):
    from twixtools import map_twix, read_twix

    print("dat_file_location is ", dat_file_location)
    if not os.path.exists(dat_file_location):
        raise FileNotFoundError("File not found")
    twixobj = read_twix(dat_file_location)[-1]

    raw_data = map_twix(twixobj)["image"]
    # raw_data.flags["remove_os"] = True # will led to warp in radial sampling
    raw_data = raw_data[:].squeeze()
    mdh = twixobj["mdb"][1].mdh
    # twixobj, mdh = mapVBVD(dat_file_location)
    # try:
    # twixobj = twixobj[-1]
    # except KeyError:
    #     self.twixobj = twixobj
    if multi_echo:
        raw_data = einx.rearrange(
            "echo par lin cha col -> echo cha par lin col",
            raw_data,
        )
        if sampling_pattern == "Cartesian":
            shape_dict = parse_shape(
                raw_data, "echo_num ch_num partition_num line_num col_num"
            )
        else:
            shape_dict = parse_shape(
                raw_data, "echo_num ch_num partition_num spoke_num spoke_len"
            )
    else:
        raw_data = einx.rearrange(
            "par lin cha col -> cha par lin col",
            raw_data,
        )
        if sampling_pattern == "Cartesian":
            shape_dict = parse_shape(raw_data, "ch_num partition_num line_num col_num")
        else:
            shape_dict = parse_shape(
                raw_data, "ch_num partition_num spoke_num spoke_len"
            )
    print(shape_dict)
    return torch.from_numpy(raw_data), shape_dict, mdh, twixobj


def check_mk_dirs(paths):
    if isinstance(paths, list):
        existance = []
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
                existance.append(False)
            else:
                existance.append(True)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)
            return False
        else:
            return True


def write_nifti(data_array, reference_nifti_file_path, output_file_path):
    """
    Write a new NIFTI file using the provided data array, reference NIFTI file's affine and header.

    Parameters:
    - data_array: numpy array, the data to be saved in the new NIFTI file.
    - reference_nifti_file_path: str, path to the reference NIFTI file.
    - output_file_path: str, path where the new NIFTI file will be saved.
    """
    # Load the reference NIFTI file
    reference_image = nib.load(reference_nifti_file_path)
    reference_image.header.set_data_dtype(np.float32)

    # Create a new NIFTI image using the data array, reference affine, and header
    new_image = nib.Nifti1Image(
        data_array, reference_image.affine, reference_image.header
    )

    # Ensure the output directory exists
    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the new NIFTI image to the specified output path
    nib.save(new_image, output_file_path)


@overload
def to_nifti(
    img: np.ndarray,
    output_path: str | bytes | os.PathLike,
    reference_nifti_path: str | bytes | os.PathLike,
):
    """
    Write a new NIFTI file using the provided data array, reference NIFTI file's affine and header.

    Parameters:
    - data_array: numpy array, the data to be saved in the new NIFTI file.
    - reference_nifti_file_path: str, path to the reference NIFTI file.
    - output_file_path: str, path where the new NIFTI file will be saved.
    """
    # Load the reference NIFTI file
    reference_image = nib.load(reference_nifti_path)
    reference_image.header.set_data_dtype(np.float32)

    # Create a new NIFTI image using the data array, reference affine, and header
    new_image = nib.Nifti1Image(img, reference_image.affine, reference_image.header)

    # Ensure the output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the new NIFTI image to the specified output path
    nib.save(new_image, output_path)

    print("Writed to: ", output_path)


@overload
def to_nifti(
    img: np.ndarray,
    output_path: str | bytes | os.PathLike,
    affine: ArrayLike,
):
    nifty_image = nib.Nifti1Image(img, affine)
    nib.save(nifty_image, output_path)
    print("Writed to: ", output_path)


@overload
def to_nifty(
    img: np.ndarray,
    output_path: str | bytes | os.PathLike,
    affine=torch.eye(4, dtype=torch.float32),
):
    nifty_image = nib.Nifti1Image(img, affine)
    # check_mk_dirs(output_path)
    nib.save(nifty_image, output_path)
    print("Writed to: ", output_path)


@overload
def to_nifti(
    img: torch.Tensor,
    output_path: str | bytes | os.PathLike,
    affine: ArrayLike,
):
    to_nifti(img.numpy(force=True), output_path, affine)


@overload
def to_nifti(
    img: DataArray,
    output_path: str | bytes | os.PathLike,
    affine: ArrayLike,
):
    to_nifti(img.to_numpy(), output_path, affine)


@dispatch
def to_nifti(img, output_path, affine):
    pass


def to_hdf5(
    img,
    output_path,
    affine=torch.eye(4, dtype=torch.float32),
    write_abs=True,
    write_complex=False,
):
    with h5py.File(output_path, "w") as f:
        if affine is not None:
            dset = f.create_dataset("affine", data=affine)
        if write_abs:
            dset = f.create_dataset("abs", data=img.abs())
        if write_complex:
            dset = f.create_dataset("imag", data=img.imag)
            dset = f.create_dataset("real", data=img.real)
    print("Writed to: ", output_path)


def from_hdf5(input_path, read_abs=True, read_complex=False):
    return_item = []
    with h5py.File(input_path, "r") as f:
        try:
            return_item.append(f["affine"][:])
        except KeyError:
            return_item.append(torch.eye(4, dtype=torch.float32))
        if read_abs:
            return_item.append(f["abs"][:])
        if read_complex:
            return_item.append(f["real"][:])
            return_item.append(f["imag"][:])
    return return_item


def to_npy():
    pass


def to_mat():
    pass


def to_analyze(img, filename, affine=None, dtype=np.float32):
    """
    the input need to be numpy array shape like w h d, the d dimension can be stack to hyperstack in ImageJ
    """
    from nibabel import analyze

    # img_analyze = eo.rearrange(img, 't ph d w h -> w h (t ph d)', t = 34, ph=5, d=72)
    # print(img_analyze.real.numpy().dtype)
    img_obj = analyze.AnalyzeImage(img.flip((1,)), affine=affine)
    img_obj.to_filename(filename)


def calculate_time(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
        # storing time before function execution
        begin = time.time()
        func(*args, **kwargs)
        # storing time after function execution
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)

    return inner1


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_cols * 5, num_rows * 5))
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = img.to("cpu")
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # plt.tight_layout()
    return _, axs


def to_dicom(
    img,
    ref_folder="/data/anlab/Chunxu/RawData_MR/CCIR_01168_ONC-DCE/ONC-DCE-004/scans",
    output_path="/data/anlab/Chunxu/RawData_MR/CCIR_01168_ONC-DCE/ONC-DCE-004/scans/MOTIF_CORD",
    dicom_header=None,
):
    fileList = glob(os.path.join(ref_folder, "*.dcm"))
    dicom_list = [pydicom.dcmread(f) for f in fileList]
    dicom_list.sort(key=lambda x: float(x.SliceLocation))

    # slicePositionList =[ float(pydicom.dcmread(f).SliceLocation) for f in fileList]
    headerStackMR = []
    for header in dicom_list:
        tempHeaderMR = deepcopy(header)
        tempHeaderMR.SeriesDescription = dicom_header["SeriesDescription"]
        tempHeaderMR.SeriesInstanceUID = pydicom.uid.generate_uid(
            entropy_srcs=dicom_header["SeriesInstanceUID"]
        )
        headerStackMR.append(tempHeaderMR)

    os.makedirs(output_path, exist_ok=True)
    img_ = img[4:76, :, :]
    k = dicom_header["FrameReferenceTime"]
    for n, header, img_slice in zip(range(img_.shape[0]), headerStackMR, img_):
        header.FrameReferenceTime = str(k)
        uid = pydicom.uid.generate_uid(
            entropy_srcs=dicom_header["SeriesInstanceUID"] + [f"contrast_{k}_slice_{n}"]
        )
        header.SOPInstanceUID = uid
        header.file_meta.MediaStorageSOPInstanceUID = uid
        header.PixelData = img_slice.tobytes()
        pydicom.dcmwrite(Path(output_path) / f"{uid}.dcm", header)


def liver_DCE_zarr_to_dicom(
    img_path_list, ref_folder, output_path, method="MCNUFFT_Contrast_34_10s"
):
    max_d = 0
    img_list = []
    for img_path in img_path_list:
        d = zarr.open(img_path, mode="r")
        current_d = np.nanpercentile(d, 99.8)
        if current_d > max_d:
            max_d = current_d
        img_list.append(d)

    for img, img_path in zip(img_list, img_path_list):
        k = Path(img_path).stem
        pid = Path(img_path).parent.stem
        header = dict(
            SeriesDescription=method,
            SeriesInstanceUID=[
                pid,
                method,
            ],  # serves as entropy source to generate UID
            FrameReferenceTime=k,
        )
        # normalize array img to the size of uint16
        img_uint16 = (np.clip(img, 0, max_d) * (4095 / max_d)).astype(np.uint16)
        to_dicom(
            np.swapaxes(np.flip(img_uint16, (0, 2)), 1, 2),
            ref_folder=ref_folder,
            output_path=output_path,
            dicom_header=header,
        )
        # break
