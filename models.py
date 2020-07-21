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

        # self.loss = 0
        # self.mse = 0
        # self.optimizer = None
        # self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        # # Build sequential layer model
        # self.activations.append(self.inputs)
        # for layer in self.layers:
        #     hidden = layer(self.activations[-1])
        #     self.activations.append(hidden)
        # self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # # Build metrics
        # self._loss()
        # self._mse()
        #
        # self.opt_op = self.optimizer.minimize(self.loss)

    def fit(self):
        pass

    def predict(self):
        pass

    # def _loss(self):
    #     raise NotImplementedError
    #
    # def _mse(self):
    #     raise NotImplementedError


class Generator(Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)

    def __call__(self, placeholders, num_features, features_nonzero, num_node, *args, **kwargs):
        self.inputs_sc = placeholders['sc_features']
        self.inputs_fc = placeholders['fc_features']
        self.input_dim = num_features
        self.node_num = num_node
        self.features_nonzero = features_nonzero
        self.adj_sc = placeholders['adj_sc']
        self.adj_fc = placeholders['adj_fc']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        self.loss = 0
        return self.predict()

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
        self.outputs = self.pred

    def _loss(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, 90*90])

        # Cross entropy error
        self.loss += tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))

    def _mse(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, 90*90])
        self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)

    def predict(self):
        return self.outputs


# def Discriminator(inputs, reuse=False):
#     with tf.variable_scope('discriminator', reuse=reuse):
#         keep_prob = 0.8
#         hidden1 = Dense(units=FLAGS.hidden4,
#                         activation=tf.nn.sigmoid)(inputs)
#         hidden2 = tf.nn.dropout(hidden1, keep_prob)
#         output = Dense(units=1,
#                        activation=tf.nn.sigmoid)(hidden2)
#     return output

class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

    def __call__(self, inputs, *args, **kwargs):
        self.inputs = inputs

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

        return self.predict()

    def _build(self):
        self.hidden1 = Dense(units=FLAGS.hidden4,
                             activation=tf.nn.sigmoid)(self.inputs)
        self.hidden2 = tf.nn.dropout(self.hidden1, keep_prob=0.8)
        self.outputs = Dense(units=1,
                            activation=tf.nn.sigmoid)(self.hidden2)

    def predict(self):
        return self.outputs



class lstmGAN():
    def __init__(self, placeholders, num_features, features_nonzero, num_node, generator, discriminator):
        self.inputs_sc = placeholders['sc_features']
        self.inputs_fc = placeholders['fc_features']
        self.input_dim = num_features
        self.node_num = num_node
        self.features_nonzero = features_nonzero
        self.adj_sc = placeholders['adj_sc']
        self.adj_fc = placeholders['adj_fc']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.generator = generator
        self.discriminator = discriminator
        self.X = tf.reshape(self.labels, [FLAGS.batch_size, self.node_num*self.node_num])

        self.G_sample = self.generator(placeholders, num_features, features_nonzero, num_node)
        self.D_fake = self.discriminator(self.G_sample)
        self.D_real = self.discriminator(self.X)

        self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake))
        preds_sub = tf.reshape(self.G_sample, [FLAGS.batch_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, 90*90])
        self.G_loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))

        self.D_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.G_loss, var_list=self.generator.vars)

