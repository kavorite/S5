from typing import Optional

import haiku as hk

from .ssm import S5SSM
from .ssm_init import make_Normal_HiPPO


def inject_step_scale(step_scale: float):
    def interceptor(f, args, kwargs, context):
        if isinstance(context.module, S5SSM) and context.method_name == "__call__":
            kwargs.update({"step_scale": step_scale})
        return f(*args, **kwargs)

    return hk.intercept_methods(interceptor)


class S5(hk.RNNCore):
    """Defines a single S5 layer, with S5 SSM, nonlinearity,
        dropout, batch/layer norm, etc. Must be vmapped.
    Args:
        ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
        dropout     (float32):  dropout rate
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                                we usually refer to this size as H
    """

    def __init__(
        self,
        width: int,
        state_width: Optional[int] = None,
        factor_rank: Optional[int] = None,
        block_count: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        liquid: bool = False,
        degree: int = 1,
        name: Optional[str] = None,
    ):
        """Initializes the ssm and dropout"""
        super().__init__(name=name)
        state_width = state_width or width
        Lambda, V = make_Normal_HiPPO(state_width, block_count)
        Vinv = V.conj().T
        BC_init = "factorized" if factor_rank is not None else "dense"
        self.width = width
        self.seq = S5SSM(
            Lambda,
            V,
            Vinv,
            width,
            state_width,
            factor_rank,
            BC_init,
            dt_min,
            dt_max,
            liquid,
            degree,
        )

    def __call__(self, x, prev_state=None, step_scale=1.0, dropout_rate=0, rng=None):
        with inject_step_scale(step_scale):
            x = self.seq(x, prev_state)
        if dropout_rate is not None:
            x = hk.dropout(rng, dropout_rate, x)
        return x
