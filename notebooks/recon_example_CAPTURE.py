# %%


import torch
from dlboost.utils.tensor_utils import complex_normalize_abs_95
from mrboost.sequence.CAPTURE_VarW_NQM_DCE_PostInj import (
    CAPTURE_VarW_NQM_DCE_PostInj_Args,
    mcnufft_reconstruct,
    preprocess_raw_data,
)


def recon_one_scan_P2P(
    raw_data, shape_dict, mdh, twixobj, phase_num=5, time_per_contrast=10
):
    # raw_data, shape_dict, mdh, twixobj = get_raw_data(dat_file_to_recon)
    args = CAPTURE_VarW_NQM_DCE_PostInj_Args(
        shape_dict,
        mdh,
        twixobj,
        phase_num=10,
        time_per_contrast=20,
        frequency_encoding_oversampling_removed=True,
        device=torch.device("cuda:1"),
    )
    data_dict_func = preprocess_raw_data(raw_data, args)
    images, csm = mcnufft_reconstruct(data_dict_func, args)
    mean, std = complex_normalize_abs_95(images, expand=False)
    images_normed = images / std
    return (
        data_dict_func["kspace_data_z"],
        data_dict_func["kspace_traj"],
        images_normed,
        csm,
        mean,
        std,
    )


# %%

recon_one_scan_P2P()

# %%
