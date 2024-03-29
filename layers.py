from inits import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="gc_weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="gcs_weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        #x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.tanh, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        x = tf.nn.relu(x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoderBatch(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.tanh, **kwargs):
        super(InnerProductDecoderBatch, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs, (0, 1, 3, 2))
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [FLAGS.batch_size, FLAGS.pred_size, -1])
        x = tf.nn.relu(x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparseWindows(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero,  batch_size, window_size, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseWindows, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot_windows(batch_size, window_size, input_dim, output_dim, name="gcsw_weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        #x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class GraphConvolutionSparseWindowsReuse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj1, adj2, features_nonzero, batch_size, window_size, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseWindowsReuse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars/', reuse=True):
            self.vars['weights'] = weight_variable_glorot_windows(batch_size, window_size, input_dim, output_dim, name="gcsw_weights")
        self.dropout = dropout
        self.adj1 = adj1
        self.adj2 = adj2
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        #x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.matmul(x, self.vars['weights'])
        x_1 = tf.matmul(self.adj1, x)
        x_2 = tf.matmul(self.adj2, x)
        outputs_1 = self.act(x_1)
        outputs_2 = self.act(x_2)
        return outputs_1, outputs_2


class GraphConvolutionSparseBatch(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero,  batch_size, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseBatch, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot_batch(batch_size, input_dim, output_dim, name="gcsb_weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        #x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductLSTM(Layer):
    def __init__(self, units, dropout=0., act=tf.nn.tanh,  **kwargs):
        super(InnerProductLSTM, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.units = units

    def _call(self, inputs):
        with tf.variable_scope(self.name, default_name='lstm') as scope:
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.units, activation=self.act)
            self.current_V, (c, h) = tf.nn.dynamic_rnn(cell=cell,
                                                       inputs=inputs,
                                                       dtype=tf.float32,
                                                       time_major=False)
            outputs = tf.reshape(h, [-1, self.units])
            c_outputs = tf.reshape(c, [-1, self.units])
            return outputs, c_outputs


class InnerProductGRU(Layer):
    def __init__(self, units, dropout=0., act=tf.nn.tanh,  **kwargs):
        super(InnerProductGRU, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.units = units

    def _call(self, inputs):
        with tf.variable_scope(self.name, default_name='lstm') as scope:
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.units, activation=self.act)
            self.current_V, h = tf.nn.dynamic_rnn(cell=cell,
                                                       inputs=inputs,
                                                       dtype=tf.float32,
                                                       time_major=False)
            outputs = tf.reshape(h, [-1, self.units])
            c_outputs = tf.reshape(h, [-1, self.units])
            return outputs, c_outputs


class InnerProduceDense(Layer):
    def __init__(self, units, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProduceDense, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        with tf.variable_scope(self.name, default_name='dense', reuse=tf.AUTO_REUSE) as scope:
            output = slim.fully_connected(inputs, self.units, self.act)
            return output


# class Attention(Layer):
#     def __init__(self, inputs_c, inputs_f, units, **kwargs):
#         super(Attention, self).__init__(**kwargs)
#         self.inputs_c = inputs_c
#         self.inputs_f = inputs_f
#         self.units = units
#         with tf.variable_scope(self.name + '_vars'):
#             self.vars['q'] = weight_variable_glorot_batch(batch_size, input_dim, output_dim, name="gcsb_weights")
#
#     def __call__(self, inputs):
#         with tf.variable_scope(self.name, default_name='att', reuse=tf.AUTO_REUSE) as scope:
