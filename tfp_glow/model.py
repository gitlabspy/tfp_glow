## mostly copied from tfp.bijectors.glow
from functools import reduce
from operator import mul
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


def create_transformed_distribution(output_shape, 
                                    num_glow_blocks=3,
                                    num_steps_per_block=32,
                                    coupling_bijector_fn=tfb.GlowDefaultNetwork,
                                    exit_bijector_fn=tfb.GlowDefaultExitNetwork,
                                    grab_after_block=None,
                                    use_actnorm=True):

    glow = tfb.Glow(output_shape=output_shape,
                    num_glow_blocks=num_glow_blocks,
                    num_steps_per_block=num_steps_per_block,
                    coupling_bijector_fn=coupling_bijector_fn,
                    exit_bijector_fn=exit_bijector_fn,
                    grab_after_block=grab_after_block,
                    use_actnorm=use_actnorm)

    transformed_distribution = tfd.TransformedDistribution(
        distribution=   tfd.MultivariateNormalDiag( tf.zeros(
                                                    (tf.math.reduce_prod(list(output_shape)), )
                                                    ),
                                                 tf.ones(
                                                    (tf.math.reduce_prod(list(output_shape)), )
                                                    )
                                               ),
        bijector    =   glow,
        name='Glow_distribution')
    return transformed_distribution
