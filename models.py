from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, GraphConvolutionSparseWindows
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from numpy import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss)

    def fit(self):
        pass

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError


class Generator(Model):
    def __init__(self, placeholders, num_features, features_nonzero, num_node, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.inputs_sc = placeholders['sc_features']
        self.inputs_fc = placeholders['fc_features']
        self.input_dim = num_features
        self.node_num = num_node
        self.features_nonzero = features_nonzero
        self.adj_sc = placeholders['adj_sc']
        self.adj_fc = placeholders['adj_fc']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.node_num,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj_sc,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs_sc)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden1,
                                                   act=tf.nn.tanh,
                                                   logging=self.logging)(self.hidden1)

        self.hidden2 = GraphConvolutionSparseWindows(input_dim=self.input_dim,
                                                     output_dim=FLAGS.hidden2,
                                                     adj=self.adj_fc,
                                                     features_nonzero=self.features_nonzero,
                                                     act=tf.nn.relu,
                                                     dropout=self.dropout,
                                                     batch_size=FLAGS.batch_size,
                                                     window_size=FLAGS.windows_size,
                                                     logging=self.logging)(self.inputs_fc)

        self.hidden2 = tf.reshape(self.hidden2, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])

        self.lstm = LSTM(input_shape=(FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2),
                         output_dim=FLAGS.hidden3,
                         activation=tf.nn.relu,
                         return_sequences=False)(self.hidden2)
        self.prefc = Dense(units=self.node_num*self.node_num,
                           activation=tf.nn.sigmoid)(self.lstm)
        self.reconstructions = tf.tile(self.reconstructions, [FLAGS.batch_size])
        self.reconstructions = tf.reshape(self.reconstructions, [FLAGS.batch_size, -1])
        self.pred = self.prefc + self.reconstructions


def Discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        keep_prob = 0.8
        hidden1 = Dense(units=FLAGS.hidden4,
                        activation=tf.nn.sigmoid)(inputs)
        hidden2 = tf.nn.dropout(hidden1, keep_prob)
        output = Dense(units=1,
                       activation=tf.nn.sigmoid)(hidden2)
    return output
