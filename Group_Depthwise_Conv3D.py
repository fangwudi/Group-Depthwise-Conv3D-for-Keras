# -*- coding: utf-8 -*-

"""
Group Conv3D in Keras, also can be used as Depthwise Conv3D.
group_multiplier: The number of convolution output channels for each group.
            The total number of output channels will be equal to `group_num * group_multiplier`.
group_size: default 1, means depthwise Conv; bigger than 1, means group Conv; input channel num should be
    integral multiple of group_size.
backend only suport tensorflow.
"""
from __future__ import absolute_import

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy.interfaces import conv3d_args_preprocessor, generate_legacy_interface
from keras.layers import Conv3D
from keras.backend.tensorflow_backend import _preprocess_padding, _preprocess_conv3d_input

import tensorflow as tf


def group_depthwise_conv3d_args_preprocessor(args, kwargs):
    converted = []

    if 'init' in kwargs:
        init = kwargs.pop('init')
        kwargs['group_depthwise_initializer'] = init
        converted.append(('init', 'group_depthwise_initializer'))

    args, kwargs, _converted = conv3d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted


legacy_group_depthwise_conv3d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=group_depthwise_conv3d_args_preprocessor)


class GroupDepthwiseConv3D(Conv3D):
    """Group Conv3D in Keras, also can be used as Depthwise Conv3D..
    # Arguments
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        group_multiplier: The number of convolution output channels for each group.
            The total number of output channels will be equal to `group_num * group_multiplier`.
        group_size: default 1, means depthwise Conv; bigger than 1, means group Conv; input channel num should be
            integral multiple of group_size.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        group_depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        group_depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        dialation_rate: List of ints.
                        Defines the dilation factor for each dimension in the
                        input. Defaults to (1,1,1)
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        group_depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        5D tensor with shape:
        `(batch, depth, channels, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(batch, filters * depth, new_depth, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters * depth)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    @legacy_group_depthwise_conv3d_support
    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 group_multiplier=1,
                 group_size=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 group_depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dilation_rate=(1, 1, 1),
                 group_depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 group_depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupDepthwiseConv3D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            dilation_rate=dilation_rate,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)

        self.group_multiplier = group_multiplier
        self.group_size = group_size
        self.group_depthwise_initializer = initializers.get(group_depthwise_initializer)
        self.group_depthwise_regularizer = regularizers.get(group_depthwise_regularizer)
        self.group_depthwise_constraint = constraints.get(group_depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.dilation_rate = dilation_rate
        self._padding = _preprocess_padding(self.padding)
        self._strides = (1,) + self.strides + (1,)
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self._data_format = "NCDHW"
        else:
            self.channel_axis = -1
            self._data_format = "NDHWC"
        self.input_dim = None
        self.kernel = None
        self.bias = None
        self.group_num = None

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `conv3d` should have rank 5. '
                             'Received input shape:', str(input_shape))
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`conv3d` '
                             'should be defined. Found `None`.')
        self.input_dim = int(input_shape[self.channel_axis])

        if self.input_dim % self.group_size != 0:
            raise ValueError('input channel num should be'
                             'integral multiple of group_size.')

        self.group_num = self.input_dim // self.group_size

        kernel_shape = (self.group_num,
                        self.kernel_size[0],
                        self.kernel_size[1],
                        self.kernel_size[2],
                        self.group_size,
                        self.group_multiplier)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.group_depthwise_initializer,
            name='kernel',
            regularizer=self.group_depthwise_regularizer,
            constraint=self.group_depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.group_num * self.group_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={self.channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs, training=None):
        inputs = _preprocess_conv3d_input(inputs, self.data_format)

        if self.data_format == 'channels_last':
            dilation = (1,) + self.dilation_rate + (1,)
        else:
            dilation = self.dilation_rate + (1,) + (1,)

        inputs = tf.split(inputs[0], self.group_num, axis=self.channel_axis)
        outputs = tf.concat(
            [tf.nn.conv3d(inp, self.kernel[i, :, :, :, :, :],
                          strides=self._strides,
                          padding=self._padding,
                          dilations=dilation,
                          data_format=self._data_format) for i, inp in enumerate(inputs)], axis=self.channel_axis)

        if self.bias is not None:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            depth = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
        else:
            depth = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]

        out_filters = self.group_num * self.group_multiplier

        depth = conv_utils.conv_output_length(depth, self.kernel_size[0],
                                              self.padding,
                                              self.strides[0])

        rows = conv_utils.conv_output_length(rows, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        cols = conv_utils.conv_output_length(cols, self.kernel_size[2],
                                             self.padding,
                                             self.strides[2])

        if self.data_format == 'channels_first':
            return input_shape[0], out_filters, depth, rows, cols

        elif self.data_format == 'channels_last':
            return input_shape[0], depth, rows, cols, out_filters

    def get_config(self):
        config = super(GroupDepthwiseConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['group_multiplier'] = self.group_multiplier
        config['group_depthwise_initializer'] = initializers.serialize(self.group_depthwise_initializer)
        config['group_depthwise_regularizer'] = regularizers.serialize(self.group_depthwise_regularizer)
        config['group_depthwise_constraint'] = constraints.serialize(self.group_depthwise_constraint)
        return config


DepthwiseConvolution3D = DepthwiseConv3D = GroupConvolution3D = GroupConv3D = GroupDepthwiseConv3D
