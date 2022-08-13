from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, uniform
from ssm_init import init_columnwise_B, init_columnwise_VinvB, init_rowwise_C, init_CV, init_VinvB, init_log_steps


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.

        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.

        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, D, input_sequence):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            D          (float32):   feedthrough matrix                   (H,)
            input_sequence (float32): input sequence of features         (L, H)
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    return jax.vmap(lambda x, u: (C_tilde @ x + D * u).real)(xs, input_sequence)


class S5SSM(nn.Module):
    Lambda_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray

    H: int
    P: int
    k: int
    BC_init: str
    discretization: str
    dt_min: float
    dt_max: float
    step_scale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_init (complex64): Initial diagonal state matrix       (P,)
            V           (complex64): Eigenvectors used for init          (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
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

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequene
        """

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda = self.param("Lambda",
                                 lambda rng, shape: self.Lambda_init,
                                 (None,))

        # Initialize input to state (B) and state to output (C) matrices
        # in basis of eigenvectors
        if self.BC_init in ["factorized"]:
            # Use a low rank factorization of rank k for B and C
            self.BH = self.param("BH",
                                 init_columnwise_B,
                                 (self.H, self.k, 2))
            self.BP = self.param("BP",
                                 lambda rng, shape: init_columnwise_VinvB(
                                                        init_columnwise_B,
                                                        rng,
                                                        shape,
                                                        self.Vinv),
                                 (self.P, self.k))

            self.CH = self.param("CH",
                                 lambda rng, shape: init_rowwise_C(rng, shape).T,
                                 (self.k, self.H, 2))
            self.CP = self.param("CP",
                                 lambda rng, shape: init_CV(init_rowwise_C,
                                                            rng, shape, self.V),
                                 (self.k, self.P, 2))

            # Parameterize complex numbers
            self.BH = self.BH[..., 0] + 1j * self.BH[..., 1]
            self.BP = self.BP[..., 0] + 1j * self.BP[..., 1]

            self.CH = self.CH[0] + 1j * self.CH[1]
            self.CP = self.CP[..., 0] + 1j * self.CP[..., 1]

            B_tilde = self.BP @ self.BH
            self.C_tilde = self.CH @ self.CP

        else:
            # Initialize B and C as dense matrices
            if self.BC_init in ["dense_columns"]:
                B_eigen_init = init_columnwise_VinvB
                B_init = init_columnwise_B
                C_init = init_rowwise_C
            elif self.BC_init in ["dense"]:
                B_eigen_init = init_VinvB
                B_init = lecun_normal()
                C_init = lecun_normal()
            else:
                raise NotImplementedError(
                       "BC_init method {} not implemented".format(self.BC_init))

            self.B = self.param("B",
                                lambda rng, shape: B_eigen_init(B_init,
                                                                rng,
                                                                shape,
                                                                self.Vinv),
                                (self.P, self.H)
                                )
            self.C = self.param("C",
                                lambda rng, shape: init_CV(C_init,
                                                           rng,
                                                           shape,
                                                           self.V),
                                (self.H, self.P, 2)
                                )

            # Parameterize complex numbers
            B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
            self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", uniform(), (self.H,))

        # Initialize learnable discretization step size
        self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
        step = self.step_scale * np.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.

        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)

        """
        return apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, self.D, input_sequence)


def init_S5SSM( H,
                P,
                k,
                Lambda_init,
                V,
                Vinv,
                BC_init,
                discretization,
                dt_min,
                dt_max
                ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM, H=H, P=P, k=k,
                   Lambda_init=Lambda_init,
                   V=V, Vinv=Vinv,
                   BC_init=BC_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max)