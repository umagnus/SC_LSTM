from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, GraphConvolutionSparseWindows, \
    InnerProductLSTM, GraphConvolutionSparseBatch, InnerProduceDense, InnerProductDecoderBatch
import tensorflow as tf
from keras.layers import Dense
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
        # self.hidden1 = GraphConvolutionSparse(input_dim=self.node_num,
        #                                       output_dim=FLAGS.hidden1,
        #                                       adj=self.adj_sc,
        #                                       features_nonzero=self.features_nonzero,
        #                                       act=tf.nn.relu,
        #                                       dropout=self.dropout,
        #                                       logging=self.logging)(self.inputs_sc)
        #
        # self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden1,
        #                                            act=tf.nn.tanh,
        #                                            logging=self.logging)(self.hidden1)

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
        # self.hidden2 = tf.concat([self.hidden2, tf.reshape(tf.tile(self.adj_sc, [FLAGS.batch_size*FLAGS.windows_size, 1]),
        #                                                    [FLAGS.batch_size, FLAGS.windows_size, -1])], 2)
        # self.hidden2.set_shape([FLAGS.batch_size, FLAGS.windows_size, self.node_num*(FLAGS.hidden2+self.node_num)])
        self.lstm = InnerProductLSTM(units=self.node_num*FLAGS.hidden3)(self.hidden2)
        self.lstm = tf.reshape(self.lstm, [FLAGS.batch_size, self.node_num, FLAGS.hidden3])
        self.adj_sc = tf.tile(self.adj_sc, [FLAGS.batch_size, 1])
        self.adj_sc = tf.reshape(self.adj_sc, [FLAGS.batch_size, self.node_num, self.node_num])
        self.hidden1 = GraphConvolutionSparseBatch(input_dim=FLAGS.hidden3,
                                                   output_dim=FLAGS.hidden1,
                                                   adj=self.adj_sc,
                                                   features_nonzero=self.features_nonzero,
                                                   batch_size=FLAGS.batch_size,
                                                   act=tf.nn.relu,
                                                   dropout=self.dropout,
                                                   logging=self.logging)(self.lstm)

        self.reconstructions = InnerProductDecoderBatch(input_dim=FLAGS.hidden1,
                                                        act=tf.nn.relu,
                                                        logging=self.logging)(self.hidden1)
        # self.hidden1 = tf.reshape(self.hidden1, [FLAGS.batch_size, -1])
        # self.reconstructions = InnerProduceDense(input_dim=self.node_num*FLAGS.hidden1,
        #                                          units=self.node_num*self.node_num,
        #                                          batch_size=FLAGS.batch_size,
        #                                          dropout=0.,
        #                                          act=tf.nn.relu)(self.hidden1)
        self.outputs = self.reconstructions
        # self.prefc = Dense(units=self.node_num*self.node_num,
        #                    activation=tf.nn.sigmoid)(self.lstm)
        # self.reconstructions = tf.tile(self.reconstructions, [FLAGS.batch_size])
        # self.reconstructions = tf.reshape(self.reconstructions, [FLAGS.batch_size, -1])
        # self.pred = self.prefc + self.reconstructions
        # self.outputs = self.pred

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


class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

    def __call__(self, placeholders, inputs, node_num, num_features, *args, **kwargs):
        self.node_num = node_num
        self.inputs = tf.reshape(inputs, [FLAGS.batch_size, self.node_num, self.node_num])
        self.fc_features = tf.reshape(placeholders['fc_features'][:, -1], [FLAGS.batch_size, self.node_num, -1])
        self.input_dim = num_features

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

        return self.predict()

    def _build(self):
        # self.hidden1 = Dense(units=FLAGS.hidden4,
        #                      activation=tf.nn.relu)(self.inputs)
        # self.hidden2 = tf.nn.dropout(self.hidden1, keep_prob=0.8)
        self.hidden1 = GraphConvolutionSparseBatch(input_dim=self.input_dim,
                                                   output_dim=FLAGS.hidden4,
                                                   adj=self.inputs,
                                                   features_nonzero=0,
                                                   batch_size=FLAGS.batch_size,
                                                   act=tf.nn.relu)(self.fc_features)
        self.hidden1 = tf.reshape(self.hidden1, [FLAGS.batch_size, -1])
        self.hidden2 = InnerProduceDense(input_dim=self.node_num*FLAGS.hidden4,
                                         units=FLAGS.hidden5,
                                         batch_size=FLAGS.batch_size,
                                         dropout=0.1,
                                         act=tf.nn.relu)(self.hidden1)
        self.hidden3 = InnerProduceDense(input_dim=FLAGS.hidden5,
                                         units=FLAGS.hidden6,
                                         batch_size=FLAGS.batch_size,
                                         dropout=0.1,
                                         act=tf.nn.relu)(self.hidden2)
        self.outputs = InnerProduceDense(input_dim=FLAGS.hidden6,
                                         units=1,
                                         batch_size=FLAGS.batch_size,
                                         act=tf.nn.sigmoid)(self.hidden3)

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
        self.D_fake = self.discriminator(placeholders, self.G_sample, num_node, num_features)
        self.D_real = self.discriminator(placeholders, self.X, num_node, num_features)

        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_real, labels=tf.ones_like(self.D_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        #self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1. - self.D_fake))
        #self.G_loss = -tf.reduce_mean(tf.log(self.D_fake))
        preds_sub = tf.reshape(self.G_sample, [FLAGS.batch_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, 90*90])
        #pre_sub = tf.reshape(self.adj_fc[:, 7], [FLAGS.batch_size, 90*90])
        self.Pre_loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))
        self.mse = tf.reduce_mean(tf.losses.mean_squared_error(preds_sub,labels_sub))

        self.D_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.G_loss, var_list=self.generator.vars)
        self.Pre_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.pre_learning_rate).minimize(self.Pre_loss, var_list=self.generator.vars)

