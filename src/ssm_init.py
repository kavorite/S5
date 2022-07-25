from jax import random
import jax.numpy as np
from jax.nn.initializers import lecun_normal
import numpy as onp


def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Note we are using original numpy instead of jax.numpy here. This
        is to make_NPLR_HIPPO matrix below.

        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    def v(n, k):
        if n > k:
            return onp.sqrt(2 * n + 1) * onp.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    mat = [[v(n, k) for k in range(1, N + 1)] for n in range(1, N + 1)]
    return -onp.array(mat)


def make_Normal_HiPPO(N):
    """ Create a normal approximation to HiPPO-LegS matrix.
         For HiPPO matrix A, A=S+pqT is normal plus low-rank for
         a certain normal matrix S and low rank terms p and q.
         We are going to approximate the HiPPO matrix with the normal matrix S.

         Note we use original numpy instead of jax.numpy first to use the
         onp.linalg.eig function. This is because Jax's linalg.eig function does not run
         on GPU for non-symmetric matrices. This creates tracing issues.
         So we instead use onp.linalg eig and then cast to a jax array
         (since we only have to do this once in the beginning to initialize).

         Args:
             N (int32): state size
         Returns:
             Lambda (complex64): eigenvalues of S (N,)
             V      (complex64): eigenvectors of S (N,N)
     """

    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * onp.sqrt(2 * onp.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, onp.newaxis] * q[onp.newaxis, :]

    # Diagonalize S to V \Lambda V^*
    Lambda, V = onp.linalg.eig(S)

    # Convert to jax array
    return np.array(Lambda), np.array(V)


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable stepsize Delta by sampling
         uniformly between
        dt_min and dt_max.

         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(key, shape):
        """ Init function

             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """ Initialize an array of learnable stepsizes

         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of log_steps (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_columnwise_B(key, shape):
    """ Initialize B matrix in columnwise fashion.
        We will sample each column of B from a lecun_normal distribution.
        This gives a different fan-in size then if we sample the entire
        matrix B at once. We found this approach to be helpful for PathX.

         Args:
             key: jax random key
             shape (tuple): desired shape, either of length 3, (P,H,_), or
                          of length 2 (N,H) depending on if the function is called
                          from the low-rank factorization initialization or a dense
                          initialization
         Returns:
             sampled B matrix (float32), either of shape (H,P) or
              shape (H,P,2) (for complex parameterization)

     """
    if len(shape) == 3:
        P, H, _ = shape
    else:
        P, H = shape

    Bs = []
    for i in range(H):
        key, skey = random.split(key)
        if len(shape) == 3:
            B = lecun_normal()(skey, shape=(P, 1, 2))
        else:
            B = lecun_normal()(skey, shape=(P, 1))
        Bs.append(B)
    return np.array(Bs)[:, :, 0]


def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then computes V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.

         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def init_columnwise_VinvB(init_fun, rng, shape, Vinv):
    """Same function as above, but with transpose applied to prevent shape mismatch
       when using the columnwise initialization. In general this is unnecessary
       and will be removed in future versions, but is left for now consistency with
       certain random seeds until we rerun experiments."""
    B = init_fun(rng, shape).T
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def init_rowwise_C(key, shape):
    """ Initialize C matrix in rowwise fashion. Analogous to init_columnwise_B function above.
        We will sample each row of C from a lecun_normal distribution.
        This gives a different fan-in size then if we sample the entire
        matrix B at once. We found this approach to be helpful for PathX.

         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return np.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """ Initialize C_tilde=BV. First samples C. Then computes CV.
        Note we will parameterize this with two different matrices for complex
        numbers.

         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)
