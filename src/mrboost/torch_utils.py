import jax
import jax.numpy as jnp
from jax import dlpack as dlpack_jax
from torch.utils import dlpack as dlpack_torch

# def unravel_index(
#     indices: torch.LongTensor,
#     shape: Tuple[int, ...],
# ) -> torch.Tensor:
#     r"""Converts flat indices into unraveled coordinates in a target shape.

#     This is a `torch` implementation of `numpy.unravel_index`.

#     Args:
#         indices: A tensor of (flat) indices, (*, N).
#         shape: The targeted shape, (D,).

#     Returns:
#         The unraveled coordinates, (*, N, D).
#     """

#     coord = []

#     for dim in reversed(shape):
#         coord.append(indices % dim)
#         indices = indices // dim

#     coord = torch.stack(coord[::-1], dim=-1)

#     return coord


# def center_crop(input_tensor, oshape):
#     """
#     Crop the input tensor to the specified output shape by centering the crop.

#     Args:
#         input_tensor (torch.Tensor): The input tensor to be cropped.
#         oshape (tuple): The desired output shape of the cropped tensor.

#     Returns:
#         torch.Tensor: The cropped tensor with the specified output shape.
#     """
#     ishape = input_tensor.shape
#     ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape, oshape)]
#     islice = tuple([slice(si, si + c) for si, c in zip(ishift, oshape)])
#     output = input_tensor[islice]
#     return output


def jax_to_torch(input):
    return dlpack_torch.from_dlpack(dlpack_jax.to_dlpack(input))


def torch_to_jax(input):
    return dlpack_jax.from_dlpack(dlpack_torch.to_dlpack(input))


@jax.jit
def as_real(x):
    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        return x

    xr = jnp.zeros(x.shape + (2,), dtype=x.real.dtype)
    xr = xr.at[..., 0].set(x.real)
    xr = xr.at[..., 1].set(x.imag)
    return xr


@jax.jit
def as_complex(x):
    assert x.shape[-1] == 2
    return jax.lax.complex(x[..., 0], x[..., 1])
