import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np 

tfk = tf.keras
tfkl = tfk.layers
tfb = tfp.bijectors


class MaskedConvLayer(tfkl.Layer):
    def __init__(self, mask_type, **kwargs):
        """
          mask_type : 'T','B','L','R' for top, bottom, left and right respectively.
        """
        super(MaskedConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(**kwargs))

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = self.conv.v.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        if self.mask_type == "T":
            self.mask[: kernel_shape[0] // 2, ...] = 1.0
        elif self.mask_type == "B":
            self.mask[(kernel_shape[-1] - kernel_shape[-1] // 2):, ...] = 1.0
        elif self.mask_type == "L":
            self.mask[:, :kernel_shape[0] // 2, ...] = 1.0
        else: # R
            self.mask[:, kernel_shape[-1] - kernel_shape[-1] // 2:, ...] = 1.0

    def call(self, inputs, is_reversed=False):
        self.conv.v.assign(self.conv.v * self.mask)
        if is_reversed:
            if self.mask_type == "B":  # rotate 180 clockwise
                self.conv.v.assign(tf.reverse(self.conv.v, [0, 1])) 
                inputs = tf.image.rot90(inputs, k=2)
                inputs = tf.image.rot90(inputs, k=2)
                self.conv.v.assign(tf.reverse(self.conv.v, [0, 1])) 
            elif self.mask_type == "L":  # rotate 90 clockwise
                self.conv.v.assign(tf.reverse(tf.transpose(self.conv.v, [1, 0, 2, 3]), [1]))
                inputs = tf.image.rot90(inputs, k=3)
                inputs = tf.image.rot90(self.conv(inputs), k=1)
                self.conv.v.assign(tf.transpose(tf.reverse(self.conv.v, [1]), [1, 0, 2, 3]))
            else: # R  rotate 270 clockwise
                self.conv.v.assign(tf.transpose(tf.reverse(self.conv.v, [1]), [1, 0, 2, 3]))
                inputs = tf.image.rot90(inputs, k=1)
                inputs = tf.image.rot90(self.conv(inputs), k=3)
                self.conv.v.assign(tf.reverse(tf.transpose(self.conv.v, [1, 0, 2, 3]), [1]))
        else:
            inputs = self.conv(inputs)
        return inputs

class MacowStep(tfb.Bijector):
    def __init__(self, this_input_shape, filters,  kernel_size=3, mask_type='T', dtype=tf.float32, validate_args=False, name=None):
        parameters = dict(locals())
        super(MacowStep, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=3,
            parameters=parameters,
            name=name or 'MacowStep')
        self.this_input_shape = this_input_shape
        self.mask_type = mask_type
        self.maskconv = tf.keras.Sequential([
            MaskedConvLayer(mask_type=mask_type, filters=filters, kernel_size=kernel_size, padding="SAME", strides=1, kernel_initializer='zeros', bias_initializer='zeros'), 
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(self.this_input_shape[-1]*2, kernel_size=1, padding="SAME", kernel_initializer='zeros', bias_initializer='zeros')),
            tf.keras.layers.ELU()
        ])
        
    def _inverse(self, y, **kwargs):
        final_tensor = tf.zeros(tf.shape(y), dtype=y.dtype)
        # _, height, width, _ =  tf.shape(y)
        height, width, channels = self.this_input_shape
        mask_constant = tf.convert_to_tensor(list(self.this_input_shape))
        if self.mask_type == 'B' or self.mask_type == 'L' or self.mask_type == 'R':
            is_reversed = True
        else:
            is_reversed = False
        
        if self.mask_type == 'L' or self.mask_type == 'R':
            def condition(h, *args):
                return h < width
            def body(h, final_tensor):
                ST_tensor = self.maskconv(final_tensor, is_reversed, **kwargs)
                mask = tf.scatter_nd(tf.stack([tf.range(height), tf.ones((width,),dtype=tf.int32)* h], axis=-1),
                        tf.ones((width, channels)),
                        mask_constant)
                mask = tf.broadcast_to(mask, final_tensor.shape)
                mu, log_scale = tf.split(ST_tensor, 2, axis=-1)
                scale = tf.math.sigmoid(log_scale + 2.)
                final_tensor = (y-mu)/(scale + 1e-12) * mask + final_tensor
                return tf.add(h, 1) ,final_tensor
        else:
            def condition(h, *args):
                return h < height
            def body(h, final_tensor):
                ST_tensor = self.maskconv(final_tensor, is_reversed, **kwargs)
                mask = tf.scatter_nd(tf.ones((1,1), dtype=tf.int32) * h, 
                        tf.ones((1, width, channels)), 
                        mask_constant)
                mask = tf.broadcast_to(mask, final_tensor.shape)
                mu, log_scale = tf.split(ST_tensor, 2, axis=-1)
                scale = tf.math.sigmoid(log_scale + 2.)
                final_tensor = (y-mu)/(scale + 1e-12) * mask + final_tensor
                return h+1 ,final_tensor
        i = tf.constant(0)
        _ , final_tensor = tf.while_loop(
            cond = condition,
            body = body,
            loop_vars = (i, final_tensor)
        )
        return final_tensor

    def _forward(self, x, **kwargs):
        ST  = self.maskconv(x , **kwargs)
        mu, log_scale = tf.split(ST, 2, axis=-1)
        scale = tf.math.sigmoid(log_scale + 2.)
        return scale * x + mu

    def _forward_log_det_jacobian(self, x, **kwargs):
        ST  = self.maskconv(x)
        mu, log_scale = tf.split(ST, 2, axis=-1)
        scale = tf.math.sigmoid(log_scale + 2.)
        return tf.reduce_sum(tf.math.log(scale), axis=[1,2,3])





