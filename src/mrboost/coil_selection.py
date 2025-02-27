from typing import Union
import torch


def energy_based_coil_selection(kspace_data, num_coils_to_select: Union[int, float]):
    """
    Perform energy-based coil selection on multi-coil k-space data.

    Args:
        k_space_data (torch.Tensor): Multi-coil k-space data of shape (num_coils, kspace_points).
        num_coils_to_select (int): Number of coils to select.

    Returns:
        torch.Tensor: Selected k-space data of shape (num_coils_to_select, kspace_points).
        torch.Tensor: Indices of the selected coils.
    """
    # Compute the energy of each coil
    coil_energy = torch.sum(torch.abs(kspace_data) ** 2, dim=-1)

    # Rank coils by energy (descending order)
    sorted_indices = torch.argsort(coil_energy, descending=True)

    if isinstance(num_coils_to_select, float):
        num_coils_to_select = int(num_coils_to_select * kspace_data.shape[0])

    # Select the top coils
    selected_indices = sorted_indices[:num_coils_to_select]

    return selected_indices


# # Example usage
# num_coils = 32
# height, width = 256, 256
# k_space_data = torch.randn(
#     num_coils, height, width
# )  # Simulated multi-coil k-space data
# num_coils_to_select = 16

# selected_k_space_data, selected_indices = energy_based_coil_selection(
#     k_space_data, num_coils_to_select
# )
# print(f"Selected coil indices: {selected_indices}")
# print(f"Shape of selected k-space data: {selected_k_space_data.shape}")
