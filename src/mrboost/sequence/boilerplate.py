from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class ReconArgs:
    shape_dict: Dict
    mdh: Any
    twixobj: Any
    device: torch.device = torch.device("cuda")
    partial_fourier_flag: bool = True
    readout_oversampling_removed: bool = False
    amplitude_scale_factor: float = field(init=False)
    ch_num: int = field(init=False)
    partition_num: int = field(init=False)
    spoke_num: int = field(init=False)
    spoke_len: int = field(init=False)
    TR: float = field(init=False)
    T: float = field(init=False)
    Fs: float = field(init=False)
    FOV: float = field(init=False)
    im_size: tuple[int, int] = field(init=False)
    kspace_centre_partition_num: Optional[int] = field(default=None, init=False)
    slice_num: int = field(init=False)

    def __post_init__(self):
        if self.partial_fourier_flag:
            for i in range(len(self.twixobj["mdb"])):
                if self.twixobj["mdb"][i].is_image_scan():
                    self.kspace_centre_partition_num = int(
                        self.twixobj["mdb"][i].mdh.CenterPar
                    )
                    break
            print(self.kspace_centre_partition_num)
            try:
                self.slice_num = round(
                    self.twixobj["hdr"]["Meas"]["lImagesPerSlab"]
                    * (1 + self.twixobj["hdr"]["Meas"]["dSliceOversamplingForDialog"])
                )
            except TypeError:
                self.slice_num = round(self.twixobj["hdr"]["Meas"]["lImagesPerSlab"])
        else:
            self.slice_num = round(self.twixobj["hdr"]["Meas"]["lImagesPerSlab"])

        # self.amplitude_scale_factor = 80 * 20 * 131072 / 65536 * 20000
        self.amplitude_scale_factor = 80 * 20 * 131072 / 65536

        self.ch_num = self.shape_dict["ch_num"]
        self.partition_num = self.shape_dict["partition_num"]
        self.spoke_num = self.shape_dict["spoke_num"]
        self.spoke_len = self.shape_dict["spoke_len"]

        self.TR = self.twixobj["hdr"]["MeasYaps"]["alTR"][0] / 1000
        self.T = (
            self.TR * self.partition_num * 1e-3 + 18.86e-3
        )  # 19e-3 is for Q-fat sat
        self.Fs = 1 / self.T
        try:
            self.FOV = self.twixobj["hdr"]["Meas"]["RoFOV"]  # type: ignore
        except KeyError:
            self.FOV = self.twixobj["hdr"]["Protocol"]["RoFOV"]
        if self.readout_oversampling_removed:
            self.im_size = (self.spoke_len, self.spoke_len)
        else:
            self.im_size = (self.spoke_len // 2, self.spoke_len // 2)
