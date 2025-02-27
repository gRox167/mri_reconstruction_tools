from mrboost.io_utils import get_raw_data
from mrboost.reconstruction import (
    CAPTURE_VarW_NQM_DCE_PostInj_Args,
    mcnufft_reconstruction,
    preprocess_raw_data,
)


def test_CAPTURE_VarW_NQM_DCE_PostInj():
    raw_data, shape_dict, mdh, twixobj = get_raw_data(
        "/bmrc-an-data/RawData_MR/CCIR_01168_ONC-DCE/ONC-DCE-001/meas_MID00144_FID02406_CAPTURE_FA15_Dyn.dat "
    )
    args = CAPTURE_VarW_NQM_DCE_PostInj_Args(
        shape_dict=shape_dict,
        mdh=mdh,
        twixobj=twixobj,
        phase_num=10,
        time_per_contrast=20,
    )
    args = preprocess_raw_data(args)
    mcnufft_reconstruction(args)
