# if os.environ("KERAS_BACKEND") == "jax":
#     from jaxtyping import Array
# elif os.environ("KERAS_BACKEND") == "torch":
#     from torch import Tensor as Array
# else:
#     from numpy import ndarray as Array
from jaxtyping import Complex, Float, Shaped
from torch import Tensor as Array

KspaceData = Complex[Array, "length"]
KspaceSpokesData = Complex[Array, "spokes_num spoke_length"]
KspaceTraj = Float[Array, "2 length"]
KspaceTraj3D = Float[Array, "3 length"]
KspaceSpokesTraj = Float[Array, "2 spokes_num spoke_length"]
KspaceSpokesTraj3D = Float[Array, "3 spokes_num spoke_length"]
Image2D = Float[Array, "h w"]
Image3D = Shaped[Image2D, "d"]
ComplexImage2D = Complex[Array, "h w"] | Image2D
ComplexImage3D = Shaped[ComplexImage2D, "d"] # d h w 
