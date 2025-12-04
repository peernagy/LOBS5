from flax import linen as nn
import jax
from typing import Any


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
            dtype       (Any):      computation dtype for Dense layers (bfloat16 or float32)
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    dtype: Any = jax.numpy.float32  # 默认 FP32，由上层传入 BF16
    use_remat: bool = False  # Gradient checkpointing: 反向时重计算激活值以节省内存

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        # 使用 nn.remat 包装 SSM 以节省内存 (反向时重计算)
        if self.use_remat:
            self.seq = nn.remat(self.ssm)(step_rescale=self.step_rescale)
        else:
            self.seq = self.ssm(step_rescale=self.step_rescale)

        # GPT风格初始化: stddev=0.02 防止梯度爆炸
        gpt_init = nn.initializers.normal(stddev=0.02)

        # Dense 层使用 dtype 控制计算精度，param_dtype 默认 FP32 (master weights)
        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model, kernel_init=gpt_init, dtype=self.dtype)
            self.out2 = nn.Dense(self.d_model, kernel_init=gpt_init, dtype=self.dtype)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model, kernel_init=gpt_init, dtype=self.dtype)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch',
                                     dtype=self.dtype)
        else:
            self.norm = nn.LayerNorm(dtype=self.dtype)

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        # remat 已在 setup() 中应用到 self.seq，直接调用 _forward
        return self._forward(x)

    def _forward(self, x):
        """Internal forward pass, may be checkpointed for memory efficiency."""
        #jax.debug.print("call x before prenorm : {}",x)

        skip = x
        if self.prenorm:
            x = self.norm(x)

        #jax.debug.print("call x before ssm : {}",x)
        x = self.seq(x)
        #jax.debug.print("call x_m after ssm : {}",x)
        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        #jax.debug.print("call x_m[0:5] after activation : {}",x[0:2][0][0:2])


        x = skip + x
        if not self.prenorm:
            x = self.norm(x)


        return x

    def __call_rnn__(self, hidden, x, d):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
            hidden : hidden state (P,)
            x (float32): input sequence (L, d_model)
            d (bool): reset signal (L,)
        Returns:
            output sequence (float32): (L, d_model)
        """
        # remat 已在 setup() 中应用到 self.seq，直接调用 _forward_rnn
        return self._forward_rnn(hidden, x, d)

    def _forward_rnn(self, hidden, x, d):
        """Internal RNN forward pass, may be checkpointed for memory efficiency."""
        #jax.debug.print("call_rnn x before prenorm : {}",x)

        skip = x
        if self.prenorm:
            x = self.norm(x)

        hidden, x = self.seq.__call_rnn__(hidden, x, d)
        #hidden, x = jax.vmap(self.seq.__call_rnn__, in_axes=(None,1,None), out_axes=1)(hidden, x, d)


        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)

        return hidden, x
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jax.numpy.zeros((batch_size,1, hidden_size), dtype=jax.numpy.complex64)