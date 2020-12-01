"""
Modified tfb.Glow for taking extra a kwarg as conditional input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import scipy

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import blockwise
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import real_nvp
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import scale_matvec_lu
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import transpose
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.util.deferred_tensor import TransformedVariable
from tensorflow_probability.python.util.seed_stream import SeedStream

tfk = tf.keras
tfkl = tfk.layers

__all__ = [
    'Glow',
    'GlowDefaultNetwork',
    'GlowDefaultExitNetwork',
    'ConditionalNetworkWithEncoder',
    'ConditionalNetwork',
]


class Glow(chain.Chain):
  def __init__(self,
               output_shape=(32, 32, 3),
               num_glow_blocks=3,
               num_steps_per_block=32,
               coupling_bijector_fn=None,
               conditional_input_shape=(32, 32, 1),
               exit_bijector_fn=None,
               grab_after_block=None,
               use_actnorm=True,
               seed=None,
               validate_args=False,
               name='glow',
               num_hidden=[512, 512],
               kernel_shape=3,
               connection_type='whole'
               ):
    """Creates the Glow bijector.

    Args:
      output_shape: A list of integers, specifying the event shape of the
        output, of the bijectors forward pass (the image).  Specified as
        [H, W, C].
        Default Value: (32, 32, 3)
      num_glow_blocks: An integer, specifying how many downsampling levels to
        include in the model. This must divide equally into both H and W,
        otherwise the bijector would not be invertible.
        Default Value: 3
      num_steps_per_block: An integer specifying how many Affine Coupling and
        1x1 convolution layers to include at each level of the spatial
        hierarchy.
        Default Value: 32 (i.e. the value used in the original glow paper).
      coupling_bijector_fn: A function which takes the argument `input_shape`
        and returns a callable neural network (e.g. a keras.Sequential). The
        network should either return a tensor with the same event shape as
        `input_shape` (this will employ additive coupling), a tensor with the
        same height and width as `input_shape` but twice the number of channels
        (this will employ affine coupling), or a bijector which takes in a
        tensor with event shape `input_shape`, and returns a tensor with shape
        `input_shape`.
      exit_bijector_fn: Similar to coupling_bijector_fn, exit_bijector_fn is
        a function which takes the argument `input_shape` and `output_chan`
        and returns a callable neural network. The neural network it returns
        should take a tensor of shape `input_shape` as the input, and return
        one of three options: A tensor with `output_chan` channels, a tensor
        with `2 * output_chan` channels, or a bijector. Additional details can
        be found in the documentation for ExitBijector.
      grab_after_block: A tuple of floats, specifying what fraction of the
        remaining channels to remove following each glow block. Glow will take
        the integer floor of this number multiplied by the remaining number of
        channels. The default is half at each spatial hierarchy.
        Default value: None (this will take out half of the channels after each
          block.
      use_actnorm: A bool deciding whether or not to use actnorm. Data-dependent
        initialization is used to initialize this layer.
        Default value: `False`
      seed: A seed to control randomness in the 1x1 convolution initialization.
        Default value: `None` (i.e., non-reproducible sampling).
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
        Default value: `False`
      name: Python `str`, name given to ops managed by this object.
        Default value: `'glow'`.
      num_hidden, kernel_shape: Affine coupling networks params.
      connection_type: How much does condition take place in affine coupling.
        Default value: 'whole'.
    """
    # Make sure that the input shape is fully defined.
    if not tensorshape_util.is_fully_defined(output_shape):
      raise ValueError('Shape must be fully defined.')
    if tensorshape_util.rank(output_shape) != 3:
      raise ValueError('Shape ndims must be 3 for images.  Your shape is'
                       '{}'.format(tensorshape_util.rank(output_shape)))

    num_glow_blocks_ = tf.get_static_value(num_glow_blocks)
    if (num_glow_blocks_ is None or
        int(num_glow_blocks_) != num_glow_blocks_ or
        num_glow_blocks_ < 1):
      raise ValueError('Argument `num_glow_blocks` must be a statically known'
                       'positive `int` (saw: {}).'.format(num_glow_blocks))
    num_glow_blocks = int(num_glow_blocks_)

    output_shape = tensorshape_util.as_list(output_shape)
    h, w, c = output_shape
    n = num_glow_blocks
    nsteps = num_steps_per_block

    # Default Glow: Half of the channels are split off after each block,
    # and after the final block, no channels are split off.
    if grab_after_block is None:
      grab_after_block = tuple([0.5] * (n - 1) + [0.])

    # Thing we know must be true: h and w are evenly divisible by 2, n times.
    # Otherwise, the squeeze bijector will not work.
    if w % 2**n != 0:
      raise ValueError('Width must be divisible by 2 at least n times.'
                       'Saw: {} % {} != 0'.format(w, 2**n))
    if h % 2**n != 0:
      raise ValueError('Height should be divisible by 2 at least n times.')
    if h // 2**n < 1:
      raise ValueError('num_glow_blocks ({0}) is too large. The image height '
                       '({1}) must be divisible by 2 no more than {2} '
                       'times.'.format(num_glow_blocks, h,
                                       int(np.log(h) / np.log(2.))))
    if w // 2**n < 1:
      raise ValueError('num_glow_blocks ({0}) is too large. The image width '
                       '({1}) must be divisible by 2 no more than {2} '
                       'times.'.format(num_glow_blocks, w,
                                       int(np.log(h) / np.log(2.))))

    # Other things we want to be true:
    # - The number of times we take must be equal to the number of glow blocks.
    if len(grab_after_block) != num_glow_blocks:
      raise ValueError('Length of grab_after_block ({0}) must match the number'
                       'of blocks ({1}).'.format(len(grab_after_block),
                                                 num_glow_blocks))

    self._blockwise_splits = self._get_blockwise_splits(output_shape,
                                                        grab_after_block[::-1])

    # Now check on the values of blockwise splits
    if any([bs[0] < 1 for bs in self._blockwise_splits]):
      first_offender = [bs[0] for bs in self._blockwise_splits].index(True)
      raise ValueError('At at least one exit, you are taking out all of your '
                       'channels, and therefore have no inputs to later blocks.'
                       ' Try setting grab_after_block to a lower value at index'
                       '{}.'.format(first_offender))

    if any(np.isclose(gab, 0) for gab in grab_after_block):
      # Special case: if specifically exiting no channels, then the exit is
      # just an identity bijector.
      pass
    elif any([bs[1] < 1 for bs in self._blockwise_splits]):
      first_offender = [bs[1] for bs in self._blockwise_splits].index(True)
      raise ValueError('At least one of your layers has < 1 output channels. '
                       'This means you set grab_at_block too small. '
                       'Try setting grab_after_block to a larger value at index'
                       '{}.'.format(first_offender))

    # Lets start to build our bijector. We assume that the distribution is 1
    # dimensional. First, lets reshape it to an image.
    glow_chain = [
        reshape.Reshape(
            event_shape_out=[h // 2**n, w // 2**n, c * 4**n],
            event_shape_in=[h * w * c])
    ]

    seedstream = SeedStream(seed=seed, salt='random_beta')

    for i in range(n):

      # This is the shape of the current tensor
      current_shape = (h // 2**n * 2**i, w // 2**n * 2**i, c * 4**(i + 1))

      # This is the shape of the input to both the glow block and exit bijector.
      this_nchan = sum(self._blockwise_splits[i][0:2])
      this_input_shape = (h // 2**n * 2**i, w // 2**n * 2**i, this_nchan) 

      glow_chain.append(invert.Invert(ExitBijector(current_shape,
                                                   self._blockwise_splits[i],
                                                   exit_bijector_fn)))
      conditional_input_shape = this_input_shape[:-1] + (1, ) # hidden hyper
      glow_block = GlowBlock(input_shape=this_input_shape,
                             num_steps=nsteps,
                             coupling_bijector_fn=coupling_bijector_fn,
                             conditional_input_shape=conditional_input_shape,
                             use_actnorm=use_actnorm,
                             seedstream=seedstream,
                             name='glowblock_{}'.format(i),
                             num_hidden=num_hidden,
                             kernel_shape=kernel_shape,
                             connection_type=connection_type)

      if self._blockwise_splits[i][2] == 0:
        # All channels are passed to the RealNVP
        glow_chain.append(glow_block)
      else:
        # Some channels are passed around the block.
        # This is done with the Blockwise bijector.
        glow_chain.append(
            blockwise.Blockwise(
                [glow_block, identity.Identity()],
                [sum(self._blockwise_splits[i][0:2]),
                 self._blockwise_splits[i][2]]))

      # Finally, lets expand the channels into spatial features.
      glow_chain.append(
          Expand(input_shape=[
              h // 2**n * 2**i,
              w // 2**n * 2**i,
              c * 4**n // 4**i,
          ]))

    glow_chain = glow_chain[::-1]
    # To finish off, we initialize the bijector with the chain we've built
    # This way, the rest of the model attributes are taken care of for us.
    super(Glow, self).__init__(
        bijectors=glow_chain, validate_args=validate_args, name=name)

  def _get_blockwise_splits(self, input_shape, grab_after_block):
    """build list of splits to give to the blockwise_bijectors.

    The list will have 3 different splits. The first element is `nleave`
    which shows how many channels will remain in the network after each exit.
    The second element is `ngrab`, which shows how many channels will be removed
    at the exit. The third is `npass`, which shows how many channels have
    already exited at a previous junction, and are therefore passed to an
    identity bijector instead of the glow block.

    Args:
      input_shape: shape of the input data
      grab_after_block: list of floats specifying what fraction of the channels
        should exit the network after each glow block.
    Returns:
      blockwise_splits: the number of channels left, taken, and passed over for
        each glow block.
    """
    blockwise_splits = []

    ngrab, nleave, npass = 0, 0, 0

    # Build backwards
    for i, frac in enumerate(reversed(grab_after_block)):
      nchan = 4**(i + 1) * input_shape[-1]
      ngrab = int((nchan - npass) * frac)
      nleave = nchan - ngrab - npass

      blockwise_splits.append([nleave, ngrab, npass])

      # update npass for the next level
      npass += ngrab
      npass *= 4

    return blockwise_splits[::-1]

  @property
  def blockwise_splits(self):
    return self._blockwise_splits


class ExitBijector(blockwise.Blockwise):
  """The spatial coupling bijector used in Glow.

  This bijector consists of a blockwise bijector of a realNVP bijector. It is
  where Glow adds a fork between points that are split off and passed to the
  base distribution, and points that are passed onward through more Glow blocks.

  For this bijector, we include spatial coupling between the part being forked
  off, and the part being passed onward. This induces a hierarchical spatial
  dependence on samples, and results in images which look better.
  """

  def __init__(self,
               input_shape,
               blockwise_splits,
               coupling_bijector_fn=None):
    """Creates the exit bijector.

    Args:
      input_shape: A list specifying the input shape to the exit bijector.
        Used in constructing the network.
      blockwise_splits: A list of integers specifying the number of channels
        exiting the model, as well as those being left in the model, and those
        bypassing the exit bijector altogether.
      coupling_bijector_fn: A function which takes the argument `input_shape`
        and returns a callable neural network (e.g. a keras Sequential). The
        network should either return a tensor with the same event shape as
        `input_shape` (this will employ additive coupling), a tensor with the
        same height and width as `input_shape` but twice the number of channels
        (this will employ affine coupling), or a bijector which takes in a
        tensor with event shape `input_shape`, and returns a tensor with shape
        `input_shape`.
    """

    nleave, ngrab, npass = blockwise_splits

    new_input_shape = input_shape[:-1]+(nleave,)
    target_output_shape = input_shape[:-1]+(ngrab,)

    # if nleave or ngrab == 0, then just use an identity for everything.
    if nleave == 0 or ngrab == 0:
      exit_layer = None
      exit_bijector_fn = None

      self.exit_layer = exit_layer
      shift_distribution = identity.Identity()

    else:
      exit_layer = coupling_bijector_fn(new_input_shape,
                                        output_chan=ngrab)
      exit_bijector_fn = self.make_bijector_fn(
          exit_layer,
          target_shape=target_output_shape,
          scale_fn=tf.exp)
      self.exit_layer = exit_layer  # For variable tracking.
      shift_distribution = real_nvp.RealNVP(
          num_masked=nleave,
          bijector_fn=exit_bijector_fn)

    super(ExitBijector, self).__init__(
        [shift_distribution, identity.Identity()], [nleave + ngrab, npass])

  @staticmethod
  def make_bijector_fn(layer, target_shape, scale_fn=tf.nn.sigmoid):

    def bijector_fn(inputs, ignored_input):
      """Decorated function to get the RealNVP bijector."""
      # Build this so we can handle a user passing a NN that returns a tensor
      # OR an NN that returns a bijector
      possible_output = layer(inputs)

      # We need to produce a bijector, but we do not know if the layer has done
      # so. We are setting this up to handle 2 possibilities:
      # 1) The layer outputs a bijector --> all is good
      # 2) The layer outputs a tensor --> we need to turn it into a bijector.
      if isinstance(possible_output, bijector.Bijector):
        output = possible_output
      elif isinstance(possible_output, tf.Tensor):
        input_shape = inputs.get_shape().as_list()
        output_shape = possible_output.get_shape().as_list()
        assert input_shape[:-1] == output_shape[:-1]
        c = input_shape[-1]

        # For layers which output a tensor, we have two possibilities:
        # 1) There are twice as many output channels as the target --> the
        #    coupling is affine, meaning there is a scale followed by a shift.
        # 2) The number of output channels equals the target --> the
        #    coupling is additive, meaning there is just a shift
        if target_shape[-1] == output_shape[-1] // 2:
          this_scale = scale.Scale(scale_fn(possible_output[..., :c] + 2.))
          this_shift = shift.Shift(possible_output[..., c:])
          output = this_shift(this_scale)
        elif target_shape[-1] == output_shape[-1]:

          output = shift.Shift(possible_output[..., :c])
        else:
          raise ValueError('Shape inconsistent with input. Expected shape'
                           '{0} or {1} but tensor was shape {2}'.format(
                               input_shape, tf.concat(
                                   [input_shape[:-1],
                                    [2 * input_shape[-1]]], 0),
                               output_shape))
      else:
        raise ValueError('Expected a bijector or a tensor, but instead got'
                         '{}'.format(possible_output.__class__))
      return output

    return bijector_fn


class GlowBlock(chain.Chain):
  def __init__(self, input_shape, num_steps, coupling_bijector_fn, conditional_input_shape,
               use_actnorm, seedstream, name='glowblock', num_hidden=[400, 400], kernel_shape=3, connection_type='whole'):

    rnvp_block = [identity.Identity()]
    this_nchan = input_shape[-1]

    for j in range(num_steps):  
      with tf.name_scope(name or 'glowblock'):
        this_layer_input_shape = input_shape[:-1] + (input_shape[-1] // 2,)
        this_layer = coupling_bijector_fn(this_layer_input_shape, 
                                          conditional_input_shape, 
                                          num_hidden=num_hidden, 
                                          kernel_shape=kernel_shape, 
                                          connection_type=connection_type)
        bijector_fn = self.make_bijector_fn(this_layer)
        this_rnvp = invert.Invert(
            real_nvp.RealNVP(this_nchan // 2, bijector_fn=bijector_fn, name='rnvp'))

        this_rnvp.coupling_bijector_layer = this_layer
        rnvp_block.append(this_rnvp)

        rnvp_block.append(
            invert.Invert(OneByOneConv(
                this_nchan, seed=seedstream(),
                dtype=dtype_util.common_dtype(this_rnvp.variables,
                                              dtype_hint=tf.float32))))

        if use_actnorm:
          rnvp_block.append(ActivationNormalization(
              this_nchan,
              dtype=dtype_util.common_dtype(this_rnvp.variables,
                                            dtype_hint=tf.float32)))
    super(GlowBlock, self).__init__(rnvp_block[::-1], name=name)

  @staticmethod
  def make_bijector_fn(layer, scale_fn=tf.nn.sigmoid):
    def bijector_fn(inputs, ignored_input, conditional_inputs):
      possible_output = layer(inputs, conditional_inputs)
      input_shape = inputs.get_shape().as_list()
      output_shape = possible_output.get_shape().as_list()
      assert input_shape[:-1] == output_shape[:-1]
      c = input_shape[-1]
      if input_shape[-1] == output_shape[-1] // 2:
        this_scale = scale.Scale(scale_fn(possible_output[..., :c] + 2.))
        this_shift = shift.Shift(possible_output[..., c:])
        output = this_shift(this_scale)
      elif input_shape[-1] == output_shape[-1]:
        output = shift.Shift(possible_output[..., :c])
      else:
        raise ValueError('Shape inconsistent with input. Expected shape'
                     '{0} or {1} but tensor was shape {2}'.format(
                         input_shape, tf.concat(
                             [input_shape[:-1],
                              [2 * input_shape[-1]]], 0),
                         output_shape))
      return output
    return bijector_fn


class OneByOneConv(scale_matvec_lu.ScaleMatvecLU):
  """The 1x1 Conv bijector used in Glow.

  This class has a convenience function which initializes the parameters
  of the bijector.
  """

  def __init__(self, event_size, seed=None, dtype=tf.float32, **kwargs):
    lower_upper, permutation = self.trainable_lu_factorization(
        event_size, seed=seed, dtype=dtype)
    super(OneByOneConv, self).__init__(lower_upper, permutation, **kwargs)

  @staticmethod
  def trainable_lu_factorization(event_size,
                                 seed=None,
                                 dtype=tf.float32,
                                 name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
      event_size = tf.convert_to_tensor(
          event_size, dtype_hint=tf.int32, name='event_size')
      random_matrix = tf.random.uniform(
          shape=[event_size, event_size],
          dtype=dtype,
          seed=seed)
      random_orthonormal = tf.linalg.qr(random_matrix)[0]
      lower_upper, permutation = tf.linalg.lu(random_orthonormal)
      lower_upper = tf.Variable(
          initial_value=lower_upper, trainable=True, name='lower_upper')
      # Initialize a non-trainable variable for the permutation indices so
      # that its value isn't re-sampled from run-to-run.
      permutation = tf.Variable(
          initial_value=permutation, trainable=False, name='permutation')
      return lower_upper, permutation

class ActivationNormalization(bijector.Bijector):
  """Bijector to implement Activation Normalization (ActNorm)."""

  def __init__(self, nchan, dtype=tf.float32, validate_args=False, name=None):
    parameters = dict(locals())

    self._initialized = tf.Variable(False, trainable=False)
    self._m = tf.Variable(tf.zeros(nchan, dtype))
    self._s = TransformedVariable(tf.ones(nchan, dtype), exp.Exp())
    self._bijector = invert.Invert(
        chain.Chain([
            scale.Scale(self._s),
            shift.Shift(self._m),
        ]))
    super(ActivationNormalization, self).__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        parameters=parameters,
        name=name or 'ActivationNormalization')

  def _inverse(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse(y, **kwargs)

  def _forward(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward(x, **kwargs)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    with tf.control_dependencies([self._maybe_init(y, inverse=True)]):
      return self._bijector.inverse_log_det_jacobian(y, 1, **kwargs)

  def _forward_log_det_jacobian(self, x, **kwargs):
    with tf.control_dependencies([self._maybe_init(x, inverse=False)]):
      return self._bijector.forward_log_det_jacobian(x, 1, **kwargs)

  def _maybe_init(self, inputs, inverse):
    """Initialize if not already initialized."""

    def _init():
      """Build the data-dependent initialization."""
      axis = prefer_static.range(prefer_static.rank(inputs) - 1)
      m = tf.math.reduce_mean(inputs, axis=axis)
      s = (
          tf.math.reduce_std(inputs, axis=axis) +
          10. * np.finfo(dtype_util.as_numpy_dtype(inputs.dtype)).eps)
      if inverse:
        s = 1 / s
        m = -m
      else:
        m = m / s
      with tf.control_dependencies([self._m.assign(m), self._s.assign(s)]):
        return self._initialized.assign(True)

    return tf.cond(self._initialized, tf.no_op, _init)


class Expand(chain.Chain):
  """A bijector to transform channels into spatial pixels."""

  def __init__(self, input_shape, block_size=2, validate_args=False, name=None):
    parameters = dict(locals())
    self._block_size = block_size
    _, h, w, c = prefer_static.split(input_shape, [-1, 1, 1, 1])
    h, w, c = h[0], w[0], c[0]
    n = self._block_size
    b = [
        reshape.Reshape(
            event_shape_out=[h * n, w * n, c // n**2],
            event_shape_in=[h, n, w, n, c // n**2]),
        transpose.Transpose(perm=[0, 3, 1, 4, 2]),
        reshape.Reshape(
            event_shape_in=[h, w, c],
            event_shape_out=[h, w, c // n**2, n, n]),
    ]
    super(Expand, self).__init__(b, name=name or 'Expand',
                                 parameters=parameters)

class ConditionalNetwork(tfk.Model):
    def __init__(self, input_shape, cond_shape, num_hidden=[400, 400], kernel_shape=3, connection_type='whole'):
        super(ConditionalNetwork, self).__init__()
        if connection_type == 'whole':
            this_nchan = input_shape[-1] + cond_shape[-1]
        elif connection_type == 'conditional':
            this_nchan = cond_shape[-1]
        else:
            this_nchan = input_shape[-1]
        conv = functools.partial(
            tfkl.Conv2D,
            padding='same',
            kernel_initializer=tf.initializers.he_normal(),
            activation='relu')
        conv_last = functools.partial(
            tfkl.Conv2D,
            padding='same',
            kernel_initializer=tf.initializers.zeros(),
            bias_initializer=tf.initializers.zeros())
        self.conv_layer = []
        for i in num_hidden[:-1]:
            self.conv_layer.append(conv(i, kernel_shape))
        self.conv_layer.append(conv(num_hidden[-1], 1))
        self.conv_layer.append(conv_last(this_nchan, kernel_shape))
        self.connection_type = connection_type
        self.one_last_conv = conv_last(input_shape[-1] * 2, kernel_shape)

    def call(self, x, cond):
        inp = x
        x = tf.concat([x, cond], axis=-1)
        out = x
        for scl in self.conv_layer:
            out = scl(out)
        if self.connection_type == 'whole':
            return  self.one_last_conv(x + out)
        elif self.connection_type == 'conditional':
            return  self.one_last_conv(cond + out)
        else:
            return  self.one_last_conv(inp + out)


class ConditionalNetworkWithEncoder(tfk.Model):
    def __init__(self, ignore_input_one=None, 
                       ignore_input_two=None, 
                       num_hidden=[400, 400], 
                       kernel_shape=3, 
                       connection_type='whole', 
                       conditional_encoder_last=1):
        super(ConditionalNetworkWithEncoder, self).__init__()
        self.num_hidden = num_hidden
        self.kernel_shape = kernel_shape
        self.connection_type = connection_type
        self.conditional_encoder_last = conditional_encoder_last
        
    def build(self,input_shape):
        conv = functools.partial(
            tfkl.Conv2D,
            padding='same',
            kernel_initializer=tf.initializers.he_normal(),
            activation='relu')
        conv_last = functools.partial(
            tfkl.Conv2D,
            padding='same',
            kernel_initializer=tf.initializers.zeros(),
            bias_initializer=tf.initializers.zeros())
        self.conv_layer = []
        for i in self.num_hidden[:-1]:
            self.conv_layer.append(conv(i, self.kernel_shape))
        self.conv_layer.append(conv(self.num_hidden[-1], 1))
        if self.connection_type == 'whole':
            self.conv_layer.append(conv_last( input_shape[-1] + self.conditional_encoder_last, self.kernel_shape, activation='softplus' ))
        elif self.connection_type == 'conditional':
            self.conv_layer.append(conv_last( input_shape[-1], self.kernel_shape, activation='softplus' ))
        else:
            self.conv_layer.append(conv_last( input_shape[-1] + self.conditional_encoder_last, self.kernel_shape , activation='softplus' ))

        self.one_last_conv = conv_last(input_shape[-1] * 2, self.kernel_shape)
        
        self.condition_encoder = tf.keras.Sequential(
              [
                  tfkl.Dense(tf.math.reduce_prod( input_shape[1:-1])//2 , activation='softplus'),
                  tfkl.Dense(tf.math.reduce_prod( input_shape[1:-1])//2 , activation='softplus'),
                  tfkl.Dense(tf.math.reduce_prod( input_shape[1:-1]) , activation='softplus'),
                  tfkl.Reshape(input_shape[1:-1] + (self.conditional_encoder_last,))
              ]
          )
    def call(self, x, cond):
        inp = x
        cond = self.condition_encoder(cond)
        x = tf.concat([x, cond], axis=-1)
        out = x
        for scl in self.conv_layer:
            out = scl(out)
        if self.connection_type == 'whole':
            return  self.one_last_conv(x + out)
        elif self.connection_type == 'conditional':
            return  self.one_last_conv(inp + out)
        else:
            return  self.one_last_conv(inp + out)




