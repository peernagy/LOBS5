from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal


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
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
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


@jax.vmap
def binary_operator_reset(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i) * (1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j) * (1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )


# --- BF16 helpers for complex matvecs ---------------------------------------
def _to_bf16_real_imag(A_complex):
    """Return (A_real_bf16, A_imag_bf16) from a complex array."""
    # A_complex assumed dtype complex64
    return A_complex.real.astype(np.bfloat16), A_complex.imag.astype(np.bfloat16)


def complex_matvec_bf16_real_x(A_complex, x_real):
    """
    Compute y = A_complex @ x_real using BF16 matvec kernels.
    A_complex: (..., M, N) complex64
    x_real:    (..., N) float32 or bfloat16-compatible real
    Returns:
      y_complex: (..., M) complex64
    Notes:
      - Uses two BF16 matvecs: real = A.real @ x, imag = A.imag @ x
      - Casts outputs back to float32 and combines into complex64.
    """
    A_re_bf, A_im_bf = _to_bf16_real_imag(A_complex)
    x_bf = x_real.astype(np.bfloat16)

    # BF16 matvecs (maps to device GEMM/GEMV kernels)
    real_bf = np.matmul(A_re_bf, x_bf)   # shape (..., M)
    imag_bf = np.matmul(A_im_bf, x_bf)   # shape (..., M)

    real_f32 = real_bf.astype(np.float32)
    imag_f32 = imag_bf.astype(np.float32)
    return real_f32 + 1j * imag_f32


def complex_matvec_bf16(A_complex, x_complex):
    """
    Compute y = A_complex @ x_complex using BF16 matvec kernels.
    A_complex: (..., M, N) complex64
    x_complex: (..., N) complex64
    Returns:
      y_complex: (..., M) complex64
    Notes:
      - Uses four BF16 matvecs (two matvecs per real/imag combination).
      - real = A_re @ x_re - A_im @ x_im
      - imag = A_re @ x_im + A_im @ x_re
    """
    A_re_bf, A_im_bf = _to_bf16_real_imag(A_complex)
    x_re_bf = x_complex.real.astype(np.bfloat16)
    x_im_bf = x_complex.imag.astype(np.bfloat16)

    rr = np.matmul(A_re_bf, x_re_bf)
    ii = np.matmul(A_im_bf, x_im_bf)
    ri = np.matmul(A_re_bf, x_im_bf)
    ir = np.matmul(A_im_bf, x_re_bf)

    real_bf = rr - ii
    imag_bf = ri + ir

    real_f32 = real_bf.astype(np.float32)
    imag_f32 = imag_bf.astype(np.float32)
    return real_f32 + 1j * imag_f32
# -----------------------------------------------------------------------------

def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        (BF16-accelerated matvecs for B_bar @ u and C_tilde @ x)
    """

    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))

    # Use BF16 matvec for B_bar @ u (u is real input). vmap over time steps.
    Bu_elements = jax.vmap(lambda u: complex_matvec_bf16_real_x(B_bar, u))(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)

    # Use BF16 matvecs for C_tilde @ x (x is complex), vmap over states in xs.
    if conj_sym:
        # compute (C_tilde @ x).real and multiply by 2
        mapped = jax.vmap(lambda x: complex_matvec_bf16(C_tilde, x).real)(xs)
        return 2 * mapped
    else:
        return jax.vmap(lambda x: complex_matvec_bf16(C_tilde, x).real)(xs)


def apply_ssm_rnn(Lambda_bar, B_bar, C_tilde, hidden, input_sequence, resets, conj_sym, bidirectional):
    """ RNN-style apply with BF16 matvecs for B_bar @ u and C_tilde @ x """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    # B_bar @ u using BF16
    Bu_elements = jax.vmap(lambda u: complex_matvec_bf16_real_x(B_bar, u))(input_sequence)

    # New: Hidden state is simply the previous Bu
    Lambda_elements = np.concatenate([
        np.ones((1, Lambda_bar.shape[0])),
        Lambda_elements,
    ])

    Bu_elements = np.concatenate([
        hidden,
        Bu_elements,
    ])

    if resets is None:
        _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    else:
        resets = np.concatenate([
            np.zeros(1),
            resets,
        ])
        _, xs, _ = jax.lax.associative_scan(binary_operator_reset, (Lambda_elements, Bu_elements, resets))

    # extract hidden state (the last state is used for the next call)
    hidden_out = xs[np.newaxis, -1]
    xs = xs[1:]

    if bidirectional:
        raise ValueError("Cannot expect a bidirectional view if doing rnn")

    # C_tilde @ x using BF16 matvecs
    if conj_sym:
        return hidden_out, jax.vmap(lambda x: 2 * complex_matvec_bf16(C_tilde, x).real)(xs)
    else:
        return hidden_out, jax.vmap(lambda x: complex_matvec_bf16(C_tilde, x).real)(xs)


class S5SSM(nn.Module):
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    V: jax.Array
    Vinv: jax.Array
    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative.
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2 * self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)
                self.C2 = self.param("C2",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                    C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

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
             input_sequence (float32/bfloat16): input sequence (L, H)
        Returns:
            output sequence (float32/bfloat16): (L, H)
        """
        # BF16 Mixed Precision: Save input dtype and cast to FP32 for SSM operations
        input_dtype = input_sequence.dtype
        input_fp32 = input_sequence.astype(np.float32)

        ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       input_fp32,
                       self.conj_sym,
                       self.bidirectional)

        Du = jax.vmap(lambda u: self.D * u)(input_fp32)
        output = ys + Du

        # BF16 Mixed Precision: Cast output back to input dtype
        return output.astype(input_dtype)

    def __call_rnn__(self, hidden, input_sequence, resets):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
             resets (bool): input sequence (L,)
        Returns:
            output sequence (float32): (L, H)
        """

        hidden, ys = apply_ssm_rnn(self.Lambda_bar,
                                   self.B_bar,
                                   self.C_tilde,
                                   hidden,
                                   input_sequence,
                                   None,
                                   self.conj_sym,
                                   self.bidirectional)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return hidden, ys + Du


def init_S5SSM(H,
               P,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   H=H,
                   P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)
