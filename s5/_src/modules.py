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


class S5Encoder(hk.Module):
    """Defines a single S5 layer, with S5 SSM, nonlinearity,
        dropout, batch/layer norm, etc.
    Args:
        ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
        dropout     (float32):  dropout rate
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                                we usually refer to this size as H
        training    (bool):     whether in training mode or not
        prenorm     (bool):     apply prenorm if true or postnorm if false
        batchnorm   (bool):     apply batchnorm if true or layernorm if false
        bn_momentum (float32):  the batchnorm momentum if batchnorm is used
        step_scale  (float32):  allows for changing the step size, e.g. after training
                                on a different resolution for the speech commands benchmark
    """

    def __init__(
        self,
        width: int,
        state_width: int,
        factor_rank: Optional[int] = None,
        dt_min=0.001,
        dt_max=0.1,
        prenorm: bool = False,
    ):
        """Initializes the ssm, batch/layer norm and dropout"""
        Lambda, V = make_Normal_HiPPO(width)
        Vinv = V.conj().T
        BC_init = ("factorized" if factor_rank is not None else "dense",)
        self.seq = S5SSM(
            Lambda, V, Vinv, width, state_width, factor_rank, BC_init, dt_min, dt_max
        )
        self.norm = hk.LayerNorm(-1, True, True)

    def __call__(self, x, timescale=1.0, dropout_rate=None, rng=None):
        """
        Compute the LxH output of S5 layer given an LxH input.

        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)

        """
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        if dropout_rate is not None:
            x = hk.dropout(rng, dropout_rate, x)
        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x
