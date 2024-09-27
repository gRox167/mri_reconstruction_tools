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
from plum import dispatch, overload
from xarray import DataArray


def get_raw_data(dat_file_location: Path):
    from twixtools import map_twix, read_twix

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
    # twixobj.image.squeeze = True
    # twixobj.image.flagRemoveOS = False
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


@overload
def to_nifty(
    img: torch.Tensor,
    output_path: str | bytes | os.PathLike,
    affine=torch.eye(4, dtype=torch.float32),
):
    # nifty_image = nib.Nifti1Image(img.numpy(), affine)
    # nib.save(nifty_image, output_path)
    # print("Writed to: ", output_path)
    to_nifty(img.numpy(force=True), output_path, affine)


@overload
def to_nifty(
    img: DataArray,
    output_path: str | bytes | os.PathLike,
    affine=torch.eye(4, dtype=torch.float32),
):
    to_nifty(img.to_numpy(), output_path, affine)


@dispatch
def to_nifty(img, output_path, affine=torch.eye(4, dtype=torch.float32)):
    nifty_image = nib.Nifti1Image(img, affine)
    nib.save(nifty_image, output_path)
    print("Writed to: ", output_path)


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


"""
    # from .twix_metadata_def import *

    # def read_protocol(
    #     datFileLocation: Path,
    #     which_scan: int = -1,
    #     variables: List = ["iNoOfFourierLines", "lPartitions", "NEcoMeas"],
    # ):
    #     offset = 0
    #     prisma_header = np.fromfile(datFileLocation, dtype=prisma_header_type, count=1)[0]
    #     offset = prisma_header.itemsize

    #     meas_list = []
    #     for k in range(prisma_header["numberOfScansInFile"]):
    #         meas = np.fromfile(datFileLocation, dtype=meas_type, count=1, offset=offset)[0]
    #         offset += 152
    #         meas_list.append(meas)

    #     measOffset, measLength = (
    #         meas_list[which_scan]["measOffset"],
    #         meas_list[which_scan]["measLength"],
    #     )
    #     offset = measOffset
    #     nProtHeaderLen = np.fromfile(datFileLocation, dtype="<u4", count=1, offset=offset)[
    #         0
    #     ]
    #     offset += 4
    #     AAA = np.fromfile(
    #         datFileLocation, dtype="<u1", count=nProtHeaderLen - 4, offset=int(offset)
    #     )
    #     AAA = "".join(map(chr, AAA))
    #     buffer = re.sub(r"\n\s*\n", "", AAA)
    #     registered_variables = parse_buffer(buffer)
    #     offset = nProtHeaderLen + measOffset

    #     return registered_variables, offset


    # def read_scan_header(datFileLocation, offset):
    #     header = np.fromfile(
    #         datFileLocation, dtype=scan_header_type, count=1, offset=offset
    #     )
    #     offset += header.itemsize
    #     return header[0], offset


    # def read_scan_meta_data(datFileLocation, currentPosition, shape_dict):
    #     # contain dict, (mdh_status, start_pos, end_pos, col_num, ch_num)
    #     scan_meta_date_list = []
    #     # first read all mdh header
    #     acqEnded = False
    #     while not acqEnded:
    #         acquisition_data, currentPosition = read_scan_header(
    #             datFileLocation, int(currentPosition)
    #         )
    #         start_pos = currentPosition
    #         DMAlength = acquisition_data["DMAlen1"] + acquisition_data["DMAlen2"] * 65536
    #         aulEvalInfoMask = acquisition_data["aulEvalInfoMask"]
    #         mdh = determine_bitfields(aulEvalInfoMask)  # ,aulEvalInfoMaskLeastSig)
    #         if mdh["MDH_SYNCDATA"]:  # TODO what is this sync data?
    #             currentPosition += DMAlength - 192  # TODO Why?
    #             continue  # TODO Why?
    #         # Acquisition ended? [To be checked in the "while" statement above.] TODO
    #         acqEnded = mdh["MDH_ACQEND"]  # BUG will not go to end
    #         if acqEnded:
    #             break

    #         ushSamplesInScan = int(acquisition_data["ushSamplesInScan"])
    #         ushUsedChannels = int(acquisition_data["ushUsedChannels"])
    #         ushLine = int(acquisition_data["ushLine"])
    #         ushPartition = int(acquisition_data["ushPartition"])
    #         ushEcho = int(acquisition_data["ushEcho"])
    #         ushKSpaceCentrePartitionNo = int(acquisition_data["ushKSpaceCentrePartitionNo"])

    #         actual_type = np.dtype([("none", "V32"), ("data", "<f4", 2 * ushSamplesInScan)])
    #         currentPosition += actual_type.itemsize * ushUsedChannels
    #         scan_meta_date_list.append(
    #             dict(
    #                 start_pos=start_pos,
    #                 line=ushLine,  # 3000
    #                 partition=ushPartition,  # 32
    #                 echo=ushEcho,  #
    #                 kspace_centre_partition_num=ushKSpaceCentrePartitionNo,
    #             )
    #         )
    #     shape_dict["spoke_len"] = ushSamplesInScan  # 640
    #     shape_dict["ch_num"] = ushUsedChannels  # 42
    #     return scan_meta_date_list, shape_dict


    # def read_actual_data(datFileLocation, col_num, offset, ch_num, dtype="<f4"):
    #     # here V32 is that we need to read some mdh head that this function don't need
    #     actual_data = np.fromfile(
    #         datFileLocation,
    #         dtype=np.dtype([("ch_header", "V32"), ("data", dtype, 2 * col_num)]),
    #         count=ch_num,
    #         offset=offset,
    #     )
    #     # axis0: ch  axis1: complex data, odd real, even imag
    #     realPart = actual_data["data"][:, ::2]
    #     imagPart = actual_data["data"][:, 1::2]
    #     complex_data = realPart + 1j * imagPart
    #     offset += actual_data.itemsize * ch_num
    #     return complex_data, offset


    # def read_navigator_kspace_data(datFileLocation, scan_meta_date_list, shape_dict):
    #     amplitudeScaleFactor = 80 * 20 * 131072 / 65536
    #     amplitudeScaleFactor = amplitudeScaleFactor * 20000

    #     spoke_len = shape_dict["spoke_len"]
    #     ch_num = shape_dict["ch_num"]
    #     spoke_num = shape_dict["spoke_num"]
    #     partition_num = shape_dict["partition_num"]
    #     echo_num = shape_dict["echo_num"]

    #     # TODO should be numOfLine be at the location of numofPE, check with Cihat this parameter
    #     # BUG should be out of while loop, we need to read the meta data first
    #     nav = np.zeros((spoke_len, spoke_num, ch_num), dtype=np.complex64)
    #     kSpaceData = np.zeros(
    #         (spoke_len, spoke_num, partition_num, ch_num, echo_num), dtype=np.complex64
    #     )
    #     # now reread to load actual data
    #     for scan_dict in tqdm(scan_meta_date_list):
    #         line = scan_dict["line"]  # 3000
    #         partition = scan_dict["partition"]  # 32
    #         echo = scan_dict["echo"]
    #
    #          was previously mdhCounter==1, but very first scan with SYNCDATA in fl3d_vibe was confusing things.
    #          if mdh['MDH_NOISEADJSCAN'] or mdh['MDH_FIRSTSCANINSLICE']: # TODO what is this condition means
    #              numberOfColumns = ushSamplesInScan
    #              numberOfChannels = ushUsedChannels
    #              if numberOfPE:

    #         # Acquisition ended? [To be checked in the "while" statement above.] TODO
    #         acqEnded = mdh['MDH_ACQEND']
    #         if acqEnded:
    #             break
    #
    #
    #         # this complex data contain ch num's complex data
    #         complex_data, currentPosition = read_actual_data(
    #             datFileLocation, spoke_len, scan_dict["start_pos"], ch_num
    #         )
    #         complex_data *= amplitudeScaleFactor

    #         if (
    #             partition == 0 and echo == 0
    #         ):  # lastRecordedNavIndex ~= currentLine why this condition need to be add?
    #             # complex data axis0 ch axis1 complex data
    #             nav[:, line, :] = np.swapaxes(complex_data, 0, 1)
    #         # if not mdh['MDH_NOISEADJSCAN']:  # TODO why? && currentEcho==1
    #         kSpaceData[:, line, partition, :, echo] = np.swapaxes(complex_data, 0, 1)

    #         # for ch in range(ch_num):

    #         #     # TODO what this chunk of code is doing?
    #         #     if line != lastSeenLine:
    #         #         ushPartition=0
    #         #         if ch+1==ch_num:
    #         #             lastSeenLine = line

    #         #     print('Recorded navigator for Line ', line+1, ', Channel ', ch+1)
    #         #     if ch+1 == ch_num:
    #         #         lastRecordedNavIndex = line

    #         # mdhCounter = mdhCounter+1 # TODO is this useful?
    #     return nav, kSpaceData


    # def process_protocol(AAA, variables):
    #     AAA = AAA.replace(' ', '')
    #     var_values = [int(re.search(
    #         '<ParamLong."'+var+'">{([0-9]+)}', AAA).group(1)) for var in variables]
    #     return var_values


    # def parse_ascconv(buffer):
    #     vararray = re.finditer(r"(?P<name>\S*)\s*=\s*(?P<value>\S*)\n", buffer)
    #     # print(vararray)
    #     mrprot = dict()
    #     for v in vararray:
    #         try:
    #             value = float(v.group("value"))
    #         except ValueError:
    #             value = v.group("value")

    #         # now split array name and index (if present)
    #         vvarray = re.finditer(r"(?P<name>\w+)(\[(?P<ix>[0-9]+)\])?", v.group("name"))

    #         currKey = []
    #         for vv in vvarray:
    #             # print(vv.group('name'))
    #             currKey.append(vv.group("name"))
    #             if vv.group("ix") is not None:
    #                 # print(vv.group('ix'))
    #                 currKey.append(vv.group("ix"))

    #         mrprot.update({tuple(currKey): value})
    #     return mrprot


    # def parse_xprot(buffer):
    #     xprot = {}
    #     tokens = re.finditer(r'<Param(?:Bool|Long|String)\."(\w+)">\s*{([^}]*)', buffer)
    #     tokensDouble = re.finditer(
    #         r'<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)', buffer
    #     )
    #     alltokens = chain(tokens, tokensDouble)

    #     for t in alltokens:
    #         # print(t.group(1))
    #         # print(t.group(2))
    #         name = t.group(1)

    #         value = re.sub(r'("*)|( *<\w*> *[^\n]*)', "", t.groups()[-1])
    #         # value = re.sub(r'\s*',' ',value)
    #         # for some bonkers reason this inserts whitespace between all the letters!
    #         # Just look for other whitespace that \s usually does.
    #         value = re.sub(r"[\t\n\r\f\v]*", "", value.strip())
    #         try:
    #             value = float(value)
    #         except ValueError:
    #             pass

    #         xprot.update({name: value})

    #     return xprot


    # def parse_buffer(buffer):
    #     print(buffer)
    #     reASCCONV = re.compile(
    #         r"### ASCCONV BEGIN[^\n]*\n(.*)\s### ASCCONV END ###", re.DOTALL
    #     )
    #     # print(f'buffer = {buffer[0:10]}')
    #     # import pdb; pdb.set_trace()

    #     ascconv = reASCCONV.search(buffer)
    #     # print(f'ascconv = {ascconv}')
    #     if ascconv is not None:
    #         prot = parse_ascconv(ascconv.group(0))
    #     else:
    #         # prot = AttrDict()
    #         prot = dict()

    #     xprot = reASCCONV.split(buffer)
    #     # print(f'xprot = {xprot[0][0:10]}')
    #     if xprot is not None:
    #         xprot = "".join([found for found in xprot])
    #         prot2 = parse_xprot(xprot)

    #         prot.update(prot2)

    #     return prot


    # def search_for_keywords_in_AAA(registered_vars, keyword):
    #     for k, v in registered_vars.items():
    #         if isinstance(k, tuple):
    #             for element in k:
    #                 if keyword in element:
    #                     print(k, v)
    #         else:
    #             if keyword in k:
    #                 print(k, v)


    # def determine_bitfields(aulEvalInfoMask):  # ,aulEvalInfoMaskLeastSig):
    #     # unpackbits need input of dtype uint8
    #     _tmp = np.frombuffer(aulEvalInfoMask, dtype="<u1")
    #     # bit order is 'little' here, because they use IEEE little endian datatype
    #     bits = np.unpackbits(_tmp, bitorder="little").astype(bool)
    #     mdh_bool_param_dict = dict()
    #     # get index from metadata definition mdhBitFields to retrive the bool value of each param
    #     for k, v in mdhBitFields.items():
    #         mdh_bool_param_dict[k] = bits[v]
    #     return mdh_bool_param_dict


    # def read_analyze_format(path):
    #     img = load(path)
    #     data_array = img.get_fdata()
    #     return img.header, data_array
    #
"""
