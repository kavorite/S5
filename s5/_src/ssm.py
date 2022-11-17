import haiku as hk
import haiku.initializers as hki
import jax
import jax.numpy as jnp
from typing import Optional

from .ssm_init import (
    init_columnwise_B,
    init_columnwise_VinvB,
    init_CV,
    init_log_steps,
    init_rowwise_C,
    init_VinvB,
)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.

    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.

    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bars, B_bars, C_tilde, D, input_sequence):
    """Compute the LxH output of discretized SSM given an LxH input.
    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        D          (float32):   feedthrough matrix                   (H,)
        input_sequence (float32): input sequence of features         (L, H)
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """

    Bu_elements = jax.vmap(lambda B_bar, u: B_bar @ u)(B_bars, input_sequence)
    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_bars, Bu_elements))
    return jax.vmap(lambda x, u: (C_tilde @ x + D * u).real)(xs, input_sequence)


def apply_ssm_liquid(Lambda_bars, B_bars, C_tilde, D, input_sequence):
    """Liquid time constant SSM á la dynamical systems given in Eq. 8 of
    https://arxiv.org/abs/2209.12951"""
    Bu_elements = jax.vmap(lambda B_bar, u: B_bar @ u)(B_bars, input_sequence)
    _, xs = jax.lax.associative_scan(
        binary_operator, (Lambda_bars + Bu_elements, Bu_elements)
    )
    return jax.vmap(lambda x, u: (C_tilde @ x + D * u).real)(xs, input_sequence)


class S5SSM(hk.RNNCore):
    def __init__(
        self,
        Lambda_init: jnp.DeviceArray,
        V: jnp.DeviceArray,
        Vinv: jnp.DeviceArray,
        h: int,
        p: int,
        k: int,
        BC_init: str,
        dt_min: float,
        dt_max: float,
        liquid: bool = False,
        degree: int = 1,
        name=None,
    ):
        """The S5 SSM
        Args:
            Lambda_init (complex64): Initial diagonal state matrix       (P,)
            V           (complex64): Eigenvectors used for init          (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
            h           (int32):     Number of features of input seq
            p           (int32):     state size
            k           (int32):     rank of low-rank factorization (if used)
            BC_init     (string):    Specifies How B and C are initialized
                        Options: [factorized: low-rank factorization,
                                dense: dense matrix drawn from Lecun_normal]
                                dense_columns: dense matrix where the columns
                                of B and the rows of C are each drawn from Lecun_normal
                                separately (i.e. different fan-in then the dense option).
                                We found this initialization to be helpful for Pathx.
            discretization: (string) Specifies discretization method
                            options: [zoh: zero-order hold method,
                                    bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        super().__init__(name=name)
        self.liquid = liquid
        self.degree = degree
        self.Lambda = hk.get_parameter(
            "Lambda",
            shape=Lambda_init.shape,
            dtype=Lambda_init.dtype,
            init=hki.Constant(Lambda_init),
        )

        # Initialize input to state (B) and state to output (C) matrices
        # in basis of eigenvectors
        if BC_init in ["factorized"]:
            # Use a low rank factorization of rank k for B and C
            BH = hk.get_parameter("BH", (h, k, 2), init=init_columnwise_B)
            BP = hk.get_parameter("BP", (p, k, 2), init=init_columnwise_B)
            CH = hk.get_parameter("CH", (k, h, 2), init=init_rowwise_C)
            CP = hk.get_parameter("CP", (k, p, 2), init=init_rowwise_C)
            # Parameterize complex numbers
            self.BH = BH[..., 0] + 1j * BH[..., 1]
            self.BP = BP[..., 0] + 1j * BP[..., 1]

            self.CH = CH[..., 0] + 1j * CH[..., 1]
            self.CP = CP[..., 0] + 1j * CP[..., 1]

            self.B_tilde = self.BP @ self.BH.T
            self.C_tilde = self.CH.T @ self.CP

        else:
            # Initialize B and C as dense matrices
            if BC_init in ["dense_columns"]:
                B_eigen_init = init_columnwise_VinvB
                B_init = init_columnwise_B
                C_init = init_rowwise_C
            elif BC_init in ["dense"]:
                B_eigen_init = init_VinvB
                lecun = hki.VarianceScaling(1.0)
                B_init = C_init = lecun
            else:
                raise NotImplementedError(f"BC_init method {BC_init} not implemented")

            self.B = hk.get_parameter("B", (p, h, 2), init=B_eigen_init(B_init, Vinv))
            self.C = hk.get_parameter("C", (h, p, 2), init=init_CV(C_init, V))
            # Parameterize complex numbers
            self.B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
            self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = hk.get_parameter("D", (h,), init=hki.RandomUniform())

        # Initialize learnable discretization step size
        self.log_step = hk.get_parameter(
            "log_step", shape=(p,), init=init_log_steps(dt_min, dt_max)
        )

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return jnp.zeros((*batch_shape, self.C_tilde[-2]))

    def __call__(
        self,
        signal,
        prev_state=None,
        *,
        step_scale=1.0,
        discretization="zoh",
    ):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.

        Args:
            input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)

        """
        # Discretize
        if discretization in ["zoh"]:
            discretize = discretize_zoh
        elif discretization in ["bilinear"]:
            discretize = discretize_bilinear
        else:
            err = f"Unknown discretization method {discretization}"
            raise NotImplementedError(err)
        step = step_scale * jnp.exp(self.log_step)
        Lambda_bar, B_bar = discretize(self.Lambda, self.B_tilde, step)
        if self.degree != 1:
            assert (
                B_bar.shape[-2] == B_bar.shape[-1]
            ), "higher-order input operators must be full-rank"
            B_bar **= self.degree
        apply_as_rnn = prev_state is not None

        if jnp.isscalar(step_scale):
            step_scale = jnp.ones(signal.shape[-2]) * step_scale
        step = step_scale[:, None] * jnp.exp(self.log_step)

        if apply_as_rnn:
            # https://arxiv.org/abs/2209.12951v1, Eq. 9
            Bu = B_bar @ signal
            if self.liquid:
                Lambda_bar += Bu
            # https://arxiv.org/abs/2208.04933v2, Eq. 2
            x = Lambda_bar @ prev_state + Bu
            y = self.C_tilde @ x + self.D * signal
            return y, x
        else:
            Lambda_bars, B_bars = jax.vmap(discretize, (None, None, 0))(
                self.Lambda, self.B_tilde, step
            )
            forward = apply_ssm_liquid if self.liquid else apply_ssm
            return forward(Lambda_bars, B_bars, self.C_tilde, self.D, signal)
