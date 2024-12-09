# if os.environ("KERAS_BACKEND") == "jax":
#     from jaxtyping import Array
# elif os.environ("KERAS_BACKEND") == "torch":
#     from torch import Tensor as Array
# else:
#     from numpy import ndarray as Array
from torch import Tensor as Array
from jaxtyping import Complex, Float, Shaped

KspaceData = Complex[Array, "length"]
KspaceSpokesData = Float[Array, "spokes_num spoke_length"]
KspaceTraj = Float[Array, "2 length"]
KspaceSpokesTraj = Float[Array, "2 spokes_num spoke_length"]
ComplexImage2D = Complex[Array, "h w"] | Float[Array, "h w"]
ComplexImage3D = Shaped[ComplexImage2D, "d"]
