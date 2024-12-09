try:
    import cupy as cp
    from cufinufft import Plan
except ImportError:
    CUFINUFFT_AVAILABLE = False

OPTS_FIELD_DECODE = {
    "gpu_method": {1: "nonuniform pts driven", 2: "shared memory"},
    "gpu_sort": {0: "no sort (GM)", 1: "sort (GM-sort)"},
    "kerevalmeth": {0: "direct eval exp(sqrt())", 1: "Horner ppval"},
    "gpu_spreadinterponly": {
        0: "NUFFT",
        1: "spread or interpolate only",
    },
}
DTYPE_R2C = {"float32": "complex64", "float64": "complex128"}


class RawCufinufftPlan:
    """Light wrapper around the guru interface of finufft."""

    def __init__(
        self,
        samples,
        shape,
        n_trans=1,
        eps=1e-6,
        **kwargs,
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans
        self._dtype = samples.dtype
        # the first element is dummy to index type 1 with 1
        # and type 2 with 2.
        self.plans = [None, None, None]
        self.grad_plan = None

        for i in [1, 2]:
            self._make_plan(i, **kwargs)
            self._set_pts(i, samples)

    @property
    def dtype(self):
        """Return the dtype (precision) of the transform."""
        try:
            return self.plans[1].dtype
        except AttributeError:
            return DTYPE_R2C[str(self._dtype)]

    def _make_plan(self, typ, **kwargs):
        self.plans[typ] = Plan(
            typ,
            self.shape,
            self.n_trans,
            self.eps,
            dtype=DTYPE_R2C[str(self._dtype)],
            **kwargs,
        )

    def _set_pts(self, typ, samples):
        plan = self.grad_plan if typ == "grad" else self.plans[typ]
        plan.setpts(
            cp.array(samples[:, 0], copy=False),
            cp.array(samples[:, 1], copy=False),
            cp.array(samples[:, 2], copy=False) if self.ndim == 3 else None,
        )

    def _destroy_plan(self, typ):
        if self.plans[typ] is not None:
            p = self.plans[typ]
            del p
            self.plans[typ] = None

    def _destroy_plan_grad(self):
        if self.grad_plan is not None:
            p = self.grad_plan
            del p
            self.grad_plan = None

    def type1(self, coeff_data, grid_data):
        """Type 1 transform. Non Uniform to Uniform."""
        return self.plans[1].execute(coeff_data, grid_data)

    def type2(self, grid_data, coeff_data):
        """Type 2 transform. Uniform to non-uniform."""
        return self.plans[2].execute(grid_data, coeff_data)

    def toggle_grad_traj(self):
        """Toggle between the gradient trajectory and the plan for type 1 transform."""
        self.plans[2], self.grad_plan = self.grad_plan, self.plans[2]
