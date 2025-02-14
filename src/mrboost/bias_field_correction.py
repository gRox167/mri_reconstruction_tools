import ants
import numpy as np
import torch


def n4_bias_field_correction_3d_complex(
    x: torch.Tensor, num_iterations: int = 50
) -> torch.Tensor:
    """
    Apply N4 bias field correction to a 3D complex PyTorch tensor using ANTsPy.

    Args:
        x (torch.Tensor): Input 3D complex tensor of shape (D, H, W)
        num_iterations (int): Number of iterations for N4 correction

    Returns:
        torch.Tensor: Bias-corrected complex tensor with same shape as input
    """
    # Extract magnitude from complex tensor
    magnitude = x.abs().cpu().numpy().astype(np.float32)

    # Convert to ANTs image (ANTs requires channel-first format)
    ants_img = ants.from_numpy(magnitude.transpose(2, 1, 0))  # ANTs uses (W,H,D) format

    # Perform N4 bias field correction
    corrected = ants.n4_bias_field_correction(
        ants_img,
        convergence={"iters": [num_iterations] * 4, "tol": 1e-7},
        shrink_factor=4,
        mask=None,
        verbose=False,
    )

    # Convert back to PyTorch tensor
    corrected_np = corrected.numpy().transpose(
        2, 1, 0
    )  # Convert back to (D,H,W) format
    corrected_tensor = torch.from_numpy(corrected_np).to(x.device)

    # Preserve phase information
    phase = torch.angle(x)
    return torch.polar(corrected_tensor, phase)
