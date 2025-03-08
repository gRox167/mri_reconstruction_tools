import os
import time
from copy import deepcopy
from glob import glob
from pathlib import Path

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
from plum import dispatch, overload
from xarray import DataArray


def torch_to_nii_direction(data):
    return einx.rearrange("d h w -> w h d", data.flip(0, 2))


def nii_to_torch_direction(data):
    return einx.rearrange("w h d -> d h w", data).flip(0, 2)


def plot_3D(
    image,
    vmin=None,
    vmax=None,
    location=(0, 0, 0),
    scale=None,
    title=None,
    show_crosshairs=False,
):
    z, y, x = location
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

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
    axes[1].imshow(image[:, y, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].title.set_text("Coronal")
    axes[2].imshow(image[:, :, x], cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].title.set_text("Sagittal")

    # Add crosshairs if requested
    if show_crosshairs:
        # Axial view (shows y-x plane)
        axes[0].axhline(y=y, color="r", linestyle="--", alpha=0.7)
        axes[0].axvline(x=x, color="r", linestyle="--", alpha=0.7)

        # Coronal view (shows z-x plane)
        axes[1].axhline(y=z, color="r", linestyle="--", alpha=0.7)
        axes[1].axvline(x=x, color="r", linestyle="--", alpha=0.7)

        # Sagittal view (shows z-y plane)
        axes[2].axhline(y=z, color="r", linestyle="--", alpha=0.7)
        axes[2].axvline(x=y, color="r", linestyle="--", alpha=0.7)
    for ax in axes:
        ax.axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    return fig, axes


def get_raw_data(dat_file_location: Path, multi_echo=False):
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
        shape_dict = parse_shape(
            raw_data, "echo_num ch_num partition_num spoke_num spoke_len"
        )
    else:
        raw_data = einx.rearrange(
            "par lin cha col -> cha par lin col",
            raw_data,
        )
        shape_dict = parse_shape(raw_data, "ch_num partition_num spoke_num spoke_len")
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
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = img.to("cpu")
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


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
