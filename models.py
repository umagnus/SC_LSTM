from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, GraphConvolutionSparseWindows, \
    InnerProductLSTM, GraphConvolutionSparseBatch, InnerProduceDense, InnerProductDecoderBatch, \
    GraphConvolutionSparseWindowsReuse, InnerProductGRU
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import Dense
from numpy import *
from utils import t_tensor_corrcoef, hsic


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
        self.labels_feature = placeholders['labels_feature']

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
        self.lstm, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3)(self.hidden2)
        self.pred_feature = slim.fully_connected(self.lstm, self.node_num*FLAGS.remove_length, activation_fn=tf.nn.sigmoid)
        self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.remove_length])
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
        self.reconstructions_reshape = tf.reshape(self.reconstructions, [FLAGS.batch_size, self.node_num, self.node_num])
        self.diag_ones = tf.matrix_set_diag(self.reconstructions_reshape, tf.ones([FLAGS.batch_size, self.node_num]))
        self.outputs = tf.reshape(self.diag_ones, [FLAGS.batch_size, -1])
        # self.outputs = self.reconstructions
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
        return self.outputs, self.pred_feature


# class Generator_new(Model):
#     def __init__(self, **kwargs):
#         super(Generator_new, self).__init__(**kwargs)
#
#     def __call__(self, placeholders, num_features, features_nonzero, num_node, *args, **kwargs):
#         self.inputs_sc = placeholders['sc_features']
#         self.inputs_fc = placeholders['fc_features']
#         self.input_dim = num_features
#         self.node_num = num_node
#         self.features_nonzero = features_nonzero
#         self.adj_sc = placeholders['adj_sc']
#         self.adj_fc = placeholders['adj_fc']
#         self.dropout = placeholders['dropout']
#         self.labels = placeholders['labels']
#         self.labels_feature = placeholders['labels_feature']
#
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
#
#         self.build()
#         self.loss = 0
#         return self.predict()
#
#     def _build(self):
#         self.hidden3 = GraphConvolutionSparseWindows(input_dim=self.node_num,
#                                                      output_dim=FLAGS.hidden2,
#                                                      adj=self.adj_fc,
#                                                      features_nonzero=self.features_nonzero,
#                                                      act=tf.nn.relu,
#                                                      dropout=self.dropout,
#                                                      batch_size=FLAGS.batch_size,
#                                                      window_size=FLAGS.windows_size,
#                                                      logging=self.logging,
#                                                      name='gen_gcn_h3')(self.adj_fc)
#         self.hidden2 = GraphConvolutionSparseWindows(input_dim=self.input_dim,
#                                                      output_dim=FLAGS.hidden2,
#                                                      adj=self.adj_fc,
#                                                      features_nonzero=self.features_nonzero,
#                                                      act=tf.nn.relu,
#                                                      dropout=self.dropout,
#                                                      batch_size=FLAGS.batch_size,
#                                                      window_size=FLAGS.windows_size,
#                                                      logging=self.logging,
#                                                      name='gen_gcn_h2')(self.inputs_fc)
#         self.hidden3 = tf.reshape(self.hidden3, [FLAGS.batch_size, FLAGS.windows_size, self.node_num * FLAGS.hidden2])
#         self.hidden2 = tf.reshape(self.hidden2, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])
#         self.lstm, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm')(self.hidden2)
#         self.lstm_fc, _ = InnerProductLSTM(units=self.node_num * FLAGS.hidden3, name='lstm_fc')(self.hidden3)
#         self.pred_feature = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.lstm)+InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.lstm_fc)
#         self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
#         self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
#                                                             [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
#                                                  self.pred_feature], axis=-1), FLAGS.pred_size)
#         self.lstm_dense = InnerProduceDense(self.node_num*FLAGS.pred_size*FLAGS.hidden3, act=tf.nn.relu, name='gen_lstm_dense')(self.lstm_fc)
#         self.lstm_dense = tf.reshape(self.lstm_dense, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden3])
#         self.adj_sc = tf.tile(self.adj_sc, [FLAGS.batch_size*FLAGS.pred_size, 1])
#         self.adj_sc = tf.reshape(self.adj_sc, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, self.node_num])
#         self.hidden1 = GraphConvolutionSparseWindows(input_dim=FLAGS.hidden3,
#                                                      output_dim=FLAGS.hidden1,
#                                                      adj=self.adj_sc,
#                                                      features_nonzero=self.features_nonzero,
#                                                      batch_size=FLAGS.batch_size,
#                                                      window_size=FLAGS.pred_size,
#                                                      act=tf.nn.relu,
#                                                      dropout=self.dropout,
#                                                      logging=self.logging,
#                                                      name='gen_gcn_h1')(self.lstm_dense)
#         self.hidden1 = tf.reshape(self.hidden1, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden1])
#         # self.reconstructions = InnerProduceDense(self.node_num*self.node_num, tf.nn.sigmoid, name='gen_rec')(self.hidden1)
#         self.reconstructions = InnerProductDecoderBatch(input_dim=FLAGS.hidden1,
#                                                         act=tf.nn.tanh,
#                                                         logging=self.logging,
#                                                         name='gen_decoder')(self.hidden1)
#         self.reconstructions_reshape = tf.reshape(self.reconstructions, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, self.node_num])
#         self.diag_ones = tf.matrix_set_diag(self.reconstructions_reshape, tf.ones([FLAGS.batch_size, FLAGS.pred_size, self.node_num]))
#         self.outputs_dense = tf.reshape(self.diag_ones, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])
#         self.corr = (tf.abs(self.corr)+self.corr)/2
#         self.outputs = tf.reshape(self.corr, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])
#         # self.outputs = tf.nn.selu(tf.multiply(self.outputs_dense, self.outputs)+self.outputs)
#         # self.outputs = tf.nn.selu(tf.multiply(self.outputs_dense, self.outputs)+self.outputs)
#         # self.outputs = FLAGS.lamda*self.outputs_dense + (1-FLAGS.lamda)*tf.reshape(self.corr,
#         #                                                                            [FLAGS.batch_size, FLAGS.pred_size,
#         #                                                                             self.node_num*self.node_num])
#
#
#     def _loss(self):
#         preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#         labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#
#         # Cross entropy error
#         self.loss += tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))
#
#     def _mse(self):
#         preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#         labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#         self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)
#
#     def predict(self):
#         return self.outputs, self.pred_feature


# class Generator_new(Model):
#     def __init__(self, **kwargs):
#         super(Generator_new, self).__init__(**kwargs)
#
#     def __call__(self, placeholders, num_features, features_nonzero, num_node, *args, **kwargs):
#         self.inputs_sc = placeholders['sc_features']
#         self.inputs_fc = placeholders['fc_features']
#         self.input_dim = num_features
#         self.node_num = num_node
#         self.features_nonzero = features_nonzero
#         self.adj_sc = placeholders['adj_sc']
#         self.adj_fc = placeholders['adj_fc']
#         self.dropout = placeholders['dropout']
#         self.labels = placeholders['labels']
#         self.labels_feature = placeholders['labels_feature']
#
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
#
#         self.build()
#         self.loss = 0
#         return self.predict()
#
#     def _build(self):
#         self.adj_sc = tf.tile(self.adj_sc, [FLAGS.batch_size*FLAGS.pred_size, 1])
#         self.adj_sc = tf.reshape(self.adj_sc, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, self.node_num])
#         self.hidden3 = GraphConvolutionSparseWindows(input_dim=self.input_dim,
#                                                      output_dim=FLAGS.hidden2,
#                                                      adj=self.adj_sc,
#                                                      features_nonzero=self.features_nonzero,
#                                                      act=tf.nn.relu,
#                                                      dropout=self.dropout,
#                                                      batch_size=FLAGS.batch_size,
#                                                      window_size=FLAGS.windows_size,
#                                                      logging=self.logging,
#                                                      name='gen_gcn_h3')(self.inputs_fc)
#         self.hidden2 = GraphConvolutionSparseWindows(input_dim=self.input_dim,
#                                                      output_dim=FLAGS.hidden2,
#                                                      adj=self.adj_fc,
#                                                      features_nonzero=self.features_nonzero,
#                                                      act=tf.nn.relu,
#                                                      dropout=self.dropout,
#                                                      batch_size=FLAGS.batch_size,
#                                                      window_size=FLAGS.windows_size,
#                                                      logging=self.logging,
#                                                      name='gen_gcn_h2')(self.inputs_fc)
#         self.hidden2_c, self.hidden3_c = GraphConvolutionSparseWindowsReuse(input_dim=self.input_dim,
#                                                                             output_dim=FLAGS.hidden2,
#                                                                             adj1=self.adj_fc,
#                                                                             adj2=self.adj_sc,
#                                                                             features_nonzero=self.features_nonzero,
#                                                                             act=tf.nn.relu,
#                                                                             dropout=self.dropout,
#                                                                             batch_size=FLAGS.batch_size,
#                                                                             window_size=FLAGS.windows_size,
#                                                                             logging=self.logging,
#                                                                             name='gen_gcn_h_c')(self.inputs_fc)
#
#         self.hidden2 = tf.reshape(self.hidden2, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden2])
#         self.reconstructions_h2 = InnerProductDecoderBatch(input_dim=FLAGS.hidden2,
#                                                            act=tf.nn.tanh,
#                                                            logging=self.logging,
#                                                            name='gen_decoder')(self.hidden2)
#         self.reconstructions_reshape_h2 = tf.reshape(self.reconstructions_h2, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, self.node_num])
#         self.hidden3 = tf.reshape(self.hidden3, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden2])
#         self.reconstructions_h3 = InnerProductDecoderBatch(input_dim=FLAGS.hidden2,
#                                                            act=tf.nn.tanh,
#                                                            logging=self.logging,
#                                                            name='gen_decoder')(self.hidden3)
#         self.reconstructions_reshape_h3 = tf.reshape(self.reconstructions_h3, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, self.node_num])
#         self.hidden2_c = tf.reshape(self.hidden2_c, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden2])
#         self.reconstructions_h2_c = InnerProductDecoderBatch(input_dim=FLAGS.hidden2,
#                                                              act=tf.nn.tanh,
#                                                              logging=self.logging,
#                                                              name='gen_decoder')(self.hidden2_c)
#         self.reconstructions_reshape_h2_c = tf.reshape(self.reconstructions_h2_c, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, self.node_num])
#         self.hidden3_c = tf.reshape(self.hidden3_c, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden2])
#         self.reconstructions_h3_c = InnerProductDecoderBatch(input_dim=FLAGS.hidden2,
#                                                              act=tf.nn.tanh,
#                                                              logging=self.logging,
#                                                              name='gen_decoder')(self.hidden3_c)
#         self.reconstructions_reshape_h3_c = tf.reshape(self.reconstructions_h3_c, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, self.node_num])
#
#         self.hidden3 = tf.reshape(self.hidden3, [FLAGS.batch_size, FLAGS.windows_size, self.node_num * FLAGS.hidden2])
#         self.hidden2 = tf.reshape(self.hidden2, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])
#         self.hidden2_c = tf.reshape(self.hidden2_c, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])
#         self.hidden3_c = tf.reshape(self.hidden3_c, [FLAGS.batch_size, FLAGS.windows_size, self.node_num * FLAGS.hidden2])
#         self.hidden_c = (self.hidden2_c+self.hidden3_c)/2
#         self.hidden2_mc = tf.multiply(self.hidden2, self.hidden_c)
#         self.hidden3_mc = tf.multiply(self.hidden3, self.hidden_c)
#         self.hidden_mc = self.hidden2_mc + self.hidden3_mc
#         self.lstm, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm')(self.hidden_mc)
#         # self.lstm, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm')(self.hidden2_mc)
#         # self.lstm_fc, _ = InnerProductLSTM(units=self.node_num * FLAGS.hidden3, name='lstm_fc')(self.hidden3_mc)
#         # self.lstm_c, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm_c')((self.hidden2_c+self.hidden3_c)/2)
#         # self.pred_feature_h2 = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_h2')(self.lstm)
#         # self.pred_feature_h3 = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_h3')(self.lstm_fc)
#         # self.pred_feature_c = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_c')(self.lstm_c)
#         # self.pred_feature = self.pred_feature_h2 + self.pred_feature_h3 #+ self.pred_feature_c
#         self.pred_feature = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.lstm)
#         self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
#         self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
#                                                             [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
#                                                  self.pred_feature], axis=-1), FLAGS.pred_size)
#         # self.lstm_dense = InnerProduceDense(self.node_num*FLAGS.pred_size*FLAGS.hidden3, act=tf.nn.relu, name='gen_lstm_dense')(self.lstm_fc)
#         # self.lstm_dense = tf.reshape(self.lstm_dense, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden3])
#         self.corr = (tf.abs(self.corr)+self.corr)/2
#         self.outputs = tf.reshape(self.corr, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])
#
#
#     def _loss(self):
#         preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#         labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#
#         self.loss_c = tf.reduce_mean(tf.losses.mean_squared_error(self.reconstructions_h2_c, self.reconstructions_h3_c))
#         # self.loss_d = hsic(self.reconstructions_h3_c, self.reconstructions_h3, 90) \
#         #               + hsic(self.reconstructions_h2_c, self.reconstructions_h2, 90)
#         # Cross entropy error
#         self.loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub)) + FLAGS.gama*self.loss_c# + FLAGS.beta*self.loss_d
#
#     def _mse(self):
#         preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#         labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
#         self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)
#
#     def predict(self):
#         return self.outputs, self.pred_feature

class Generator_new(Model):
    def __init__(self, **kwargs):
        super(Generator_new, self).__init__(**kwargs)

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
        self.labels_feature = placeholders['labels_feature']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        self.loss = 0
        return self.predict()

    def _build(self):
        self.adj_sc = tf.tile(self.adj_sc, [FLAGS.batch_size*FLAGS.windows_size, 1])
        self.adj_sc = tf.reshape(self.adj_sc, [FLAGS.batch_size, FLAGS.windows_size, self.node_num, self.node_num])
        self.inputs_fc_new = self.inputs_fc[:, :, :, -1:]
        self.hidden3 = GraphConvolutionSparseWindows(input_dim=1,
                                                     output_dim=FLAGS.hidden2,
                                                     adj=self.adj_sc,
                                                     features_nonzero=self.features_nonzero,
                                                     act=tf.nn.relu,
                                                     dropout=self.dropout,
                                                     batch_size=FLAGS.batch_size,
                                                     window_size=FLAGS.windows_size,
                                                     logging=self.logging,
                                                     name='gen_gcn_h3')(self.inputs_fc_new)
        self.hidden2 = GraphConvolutionSparseWindows(input_dim=self.input_dim,
                                                     output_dim=FLAGS.hidden2,
                                                     adj=self.adj_fc,
                                                     features_nonzero=self.features_nonzero,
                                                     act=tf.nn.relu,
                                                     dropout=self.dropout,
                                                     batch_size=FLAGS.batch_size,
                                                     window_size=FLAGS.windows_size,
                                                     logging=self.logging,
                                                     name='gen_gcn_h2')(self.inputs_fc)
        self.hidden2_c, self.hidden3_c = GraphConvolutionSparseWindowsReuse(input_dim=self.input_dim,
                                                                            output_dim=FLAGS.hidden2,
                                                                            adj1=self.adj_fc,
                                                                            adj2=self.adj_sc,
                                                                            features_nonzero=self.features_nonzero,
                                                                            act=tf.nn.relu,
                                                                            dropout=self.dropout,
                                                                            batch_size=FLAGS.batch_size,
                                                                            window_size=FLAGS.windows_size,
                                                                            logging=self.logging,
                                                                            name='gen_gcn_h_c')(self.inputs_fc)
        self.hidden3 = tf.reshape(self.hidden3, [FLAGS.batch_size, FLAGS.windows_size, self.node_num * FLAGS.hidden2])
        self.hidden2 = tf.reshape(self.hidden2, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])
        self.hidden2_c = tf.reshape(self.hidden2_c, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])
        self.hidden3_c = tf.reshape(self.hidden3_c, [FLAGS.batch_size, FLAGS.windows_size, self.node_num * FLAGS.hidden2])
        self.hidden_c = (self.hidden2_c+self.hidden3_c)/2
        self.hidden2_mc = tf.multiply(self.hidden2, self.hidden_c)
        self.hidden3_mc = tf.multiply(self.hidden3, self.hidden_c)
        self.hidden_mc = self.hidden2_mc + self.hidden3_mc
        self.lstm, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm')(self.hidden_mc)
        # self.lstm, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm')(self.hidden2_mc)
        # self.lstm_fc, _ = InnerProductLSTM(units=self.node_num * FLAGS.hidden3, name='lstm_fc')(self.hidden3_mc)
        # self.lstm_c, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm_c')((self.hidden2_c+self.hidden3_c)/2)
        # self.pred_feature_h2 = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_h2')(self.lstm)
        # self.pred_feature_h3 = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_h3')(self.lstm_fc)
        # self.pred_feature_c = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_c')(self.lstm_c)
        # self.pred_feature = self.pred_feature_h2 + self.pred_feature_h3 #+ self.pred_feature_c
        self.pred_feature = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.lstm)
        self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
        self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
                                                            [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
                                                 self.pred_feature], axis=-1), FLAGS.pred_size)
        # self.lstm_dense = InnerProduceDense(self.node_num*FLAGS.pred_size*FLAGS.hidden3, act=tf.nn.relu, name='gen_lstm_dense')(self.lstm_fc)
        # self.lstm_dense = tf.reshape(self.lstm_dense, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden3])
        self.corr_abs = (tf.abs(self.corr)+self.corr)/2
        self.outputs = tf.reshape(self.corr_abs, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])


    def _loss(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])

        self.loss_c = tf.reduce_mean(tf.losses.mean_squared_error(self.hidden2_c1, self.hidden3_c1))
        # self.loss_d = hsic(self.reconstructions_h3_c, self.reconstructions_h3, 90) \
        #               + hsic(self.reconstructions_h2_c, self.reconstructions_h2, 90)
        # Cross entropy error
        self.loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub)) + FLAGS.gama*self.loss_c# + FLAGS.beta*self.loss_d

    def _mse(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)

    def predict(self):
        return self.outputs, self.pred_feature


class Generator_tgcn_sc(Model):
    def __init__(self, **kwargs):
        super(Generator_tgcn_sc, self).__init__(**kwargs)

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
        self.labels_feature = placeholders['labels_feature']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        self.loss = 0
        return self.predict()

    def _build(self):
        self.adj_sc = tf.tile(self.adj_sc, [FLAGS.batch_size*FLAGS.windows_size, 1])
        self.adj_sc = tf.reshape(self.adj_sc, [FLAGS.batch_size, FLAGS.windows_size, self.node_num, self.node_num])
        self.hidden3 = GraphConvolutionSparseWindows(input_dim=self.input_dim,
                                                     output_dim=FLAGS.hidden2,
                                                     adj=self.adj_sc,
                                                     features_nonzero=self.features_nonzero,
                                                     act=tf.nn.relu,
                                                     dropout=self.dropout,
                                                     batch_size=FLAGS.batch_size,
                                                     window_size=FLAGS.windows_size,
                                                     logging=self.logging,
                                                     name='gen_gcn_h3')(self.inputs_fc)
        self.hidden3 = tf.reshape(self.hidden3, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])
        self.gru, _ = InnerProductGRU(units=self.node_num*FLAGS.hidden3, name='gru_sc')(self.hidden3)
        # self.lstm, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm')(self.hidden2_mc)
        # self.lstm_fc, _ = InnerProductLSTM(units=self.node_num * FLAGS.hidden3, name='lstm_fc')(self.hidden3_mc)
        # self.lstm_c, _ = InnerProductLSTM(units=self.node_num*FLAGS.hidden3, name='lstm_c')((self.hidden2_c+self.hidden3_c)/2)
        # self.pred_feature_h2 = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_h2')(self.lstm)
        # self.pred_feature_h3 = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_h3')(self.lstm_fc)
        # self.pred_feature_c = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense_c')(self.lstm_c)
        # self.pred_feature = self.pred_feature_h2 + self.pred_feature_h3 #+ self.pred_feature_c
        self.pred_feature = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.gru)
        self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
        self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
                                                            [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
                                                 self.pred_feature], axis=-1), FLAGS.pred_size)
        # self.lstm_dense = InnerProduceDense(self.node_num*FLAGS.pred_size*FLAGS.hidden3, act=tf.nn.relu, name='gen_lstm_dense')(self.lstm_fc)
        # self.lstm_dense = tf.reshape(self.lstm_dense, [FLAGS.batch_size, FLAGS.pred_size, self.node_num, FLAGS.hidden3])
        self.corr_abs = (tf.abs(self.corr)+self.corr)/2
        self.outputs = tf.reshape(self.corr_abs, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])


    def _loss(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])

        # self.loss_d = hsic(self.reconstructions_h3_c, self.reconstructions_h3, 90) \
        #               + hsic(self.reconstructions_h2_c, self.reconstructions_h2, 90)
        # Cross entropy error
        self.loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))

    def _mse(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)

    def predict(self):
        return self.outputs, self.pred_feature


class Generator_tgcn_fc(Model):
    def __init__(self, **kwargs):
        super(Generator_tgcn_fc, self).__init__(**kwargs)

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
        self.labels_feature = placeholders['labels_feature']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        self.loss = 0
        return self.predict()

    def _build(self):
        self.hidden3 = GraphConvolutionSparseWindows(input_dim=self.input_dim,
                                                     output_dim=FLAGS.hidden2,
                                                     adj=self.adj_fc,
                                                     features_nonzero=self.features_nonzero,
                                                     act=tf.nn.relu,
                                                     dropout=self.dropout,
                                                     batch_size=FLAGS.batch_size,
                                                     window_size=FLAGS.windows_size,
                                                     logging=self.logging,
                                                     name='gen_gcn_h3')(self.inputs_fc)
        self.hidden3 = tf.reshape(self.hidden3, [FLAGS.batch_size, FLAGS.windows_size, self.node_num*FLAGS.hidden2])
        self.gru, _ = InnerProductGRU(units=self.node_num*FLAGS.hidden3, name='gru_fc')(self.hidden3)
        self.pred_feature = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.gru)
        self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
        self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
                                                            [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
                                                 self.pred_feature], axis=-1), FLAGS.pred_size)
        self.corr_abs = (tf.abs(self.corr)+self.corr)/2
        self.outputs = tf.reshape(self.corr_abs, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])


    def _loss(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])

        # Cross entropy error
        self.loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))

    def _mse(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)

    def predict(self):
        return self.outputs, self.pred_feature


class Generator_fnn(Model):
    def __init__(self, **kwargs):
        super(Generator_fnn, self).__init__(**kwargs)

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
        self.labels_feature = placeholders['labels_feature']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        self.loss = 0
        return self.predict()

    def _build(self):
        self.fc_bg = tf.reshape(self.inputs_fc[:, 0, :], [FLAGS.batch_size, self.node_num, FLAGS.window_length])
        self.fc_ed = tf.reshape(self.inputs_fc[:, FLAGS.windows_size-1, :, FLAGS.window_length-FLAGS.windows_size:], [FLAGS.batch_size, self.node_num, FLAGS.window_length-FLAGS.windows_size])
        self.inputs = tf.concat([self.fc_bg, self.fc_ed], axis=2)
        self.inputs = tf.reshape(self.inputs, [FLAGS.batch_size, self.node_num*(FLAGS.window_length+FLAGS.windows_size)])
        self.hidden_1 = InnerProduceDense(self.node_num*80, act=tf.nn.sigmoid, name='gen_h1')(self.inputs)
        self.hidden_2 = InnerProduceDense(self.node_num * 60, act=tf.nn.sigmoid, name='gen_h2')(self.hidden_1)
        self.pred_feature = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.hidden_2)
        self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
        self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
                                                            [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
                                                 self.pred_feature], axis=-1), FLAGS.pred_size)
        self.corr_abs = (tf.abs(self.corr)+self.corr)/2
        self.outputs = tf.reshape(self.corr_abs, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])

    def _loss(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])

        # self.loss_d = hsic(self.reconstructions_h3_c, self.reconstructions_h3, 90) \
        #               + hsic(self.reconstructions_h2_c, self.reconstructions_h2, 90)
        # Cross entropy error
        self.loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))

    def _mse(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)

    def predict(self):
        return self.outputs, self.pred_feature


class Generator_HA(Model):
    def __init__(self, **kwargs):
        super(Generator_HA, self).__init__(**kwargs)

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
        self.labels_feature = placeholders['labels_feature']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        self.loss = 0
        return self.predict()

    def _build(self):
        self.fc_bg = tf.reshape(self.inputs_fc[:, 0, :], [FLAGS.batch_size, self.node_num, FLAGS.window_length])
        self.fc_ed = tf.reshape(self.inputs_fc[:, FLAGS.windows_size-1, :, FLAGS.window_length-FLAGS.windows_size:], [FLAGS.batch_size, self.node_num, FLAGS.window_length-FLAGS.windows_size])
        self.inputs = tf.concat([self.fc_bg, self.fc_ed], axis=2)
        self.pred_feature = self.inputs[:, :, -1]
        self.pred_feature = tf.tile(self.pred_feature, [1,  FLAGS.pred_size])
        self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
        self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
                                                            [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
                                                 self.pred_feature], axis=-1), FLAGS.pred_size)
        self.corr_abs = (tf.abs(self.corr)+self.corr)/2
        self.outputs = tf.reshape(self.corr_abs, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])

    def _loss(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])

        # self.loss_d = hsic(self.reconstructions_h3_c, self.reconstructions_h3, 90) \
        #               + hsic(self.reconstructions_h2_c, self.reconstructions_h2, 90)
        # Cross entropy error
        self.loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub))

    def _mse(self):
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        self.mse = tf.losses.mean_squared_error(preds_sub, labels_sub)

    def predict(self):
        return self.outputs, self.pred_feature


class Discriminator_old(Model):
    def __init__(self, **kwargs):
        super(Discriminator_old, self).__init__(**kwargs)

    def __call__(self, placeholders, inputs, node_num, num_features, *args, **kwargs):
        self.node_num = node_num
        self.inputs = tf.reshape(inputs, [FLAGS.batch_size, self.node_num, self.node_num])
        # self.inputs = inputs
        self.fc_features = tf.reshape(placeholders['fc_features'][:, -1], [FLAGS.batch_size, self.node_num, -1])
        self.input_dim = num_features

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

        return self.predict()

    def _build(self):
        # self.hidden1 = GraphConvolutionSparseBatch(input_dim=self.input_dim,
        #                                            output_dim=FLAGS.hidden4,
        #                                            adj=self.inputs,
        #                                            features_nonzero=0,
        #                                            batch_size=FLAGS.batch_size,
        #                                            act=tf.nn.relu)(self.fc_features)
        # self.hidden1 = tf.reshape(self.hidden1, [FLAGS.batch_size, -1])
        reuse = len([t for t in tf.global_variables() if t.name.startswith(self.name)]) > 0
        with tf.variable_scope(self.name, reuse=reuse):
            self.hidden1 = GraphConvolutionSparseBatch(input_dim=self.input_dim,
                                                       output_dim=FLAGS.hidden4,
                                                       adj=self.inputs,
                                                       features_nonzero=0,
                                                       batch_size=FLAGS.batch_size,
                                                       act=tf.nn.relu)(tf.nn.l2_normalize(self.fc_features, dim=[1, 2]))
            self.hidden1 = tf.reshape(self.hidden1, [FLAGS.batch_size, -1])
            # self.hidden1 = slim.fully_connected(self.inputs, self.node_num*FLAGS.hidden4, activation_fn=tf.nn.relu)
            self.hidden2 = slim.fully_connected(self.hidden1, FLAGS.hidden5, activation_fn=tf.nn.relu)
            self.hidden3 = slim.fully_connected(self.hidden2, FLAGS.hidden6, activation_fn=tf.nn.relu)
            self.outputs = slim.fully_connected(self.hidden3, 1, activation_fn=None)
        # self.hidden1 = InnerProduceDense(input_dim=self.node_num*self.node_num,
        #                                  units=self.node_num*FLAGS.hidden4,
        #                                  batch_size=FLAGS.batch_size,
        #                                  dropout=0.1,
        #                                  act=tf.nn.leaky_relu)(self.inputs)
        # self.hidden2 = InnerProduceDense(input_dim=self.node_num*FLAGS.hidden4,
        #                                  units=FLAGS.hidden5,
        #                                  batch_size=FLAGS.batch_size,
        #                                  dropout=0.1,
        #                                  act=tf.nn.leaky_relu)(self.hidden1)
        # self.hidden3 = InnerProduceDense(input_dim=FLAGS.hidden5,
        #                                  units=FLAGS.hidden6,
        #                                  batch_size=FLAGS.batch_size,
        #                                  dropout=0.1,
        #                                  act=tf.nn.leaky_relu)(self.hidden2)
        # self.outputs = InnerProduceDense(input_dim=FLAGS.hidden6,
        #                                  units=1,
        #                                  batch_size=FLAGS.batch_size,
        #                                  act=lambda x: x)(self.hidden3)

    def predict(self):
        return self.outputs


class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

    def __call__(self, placeholders, inputs, node_num, num_features, *args, **kwargs):
        self.node_num = node_num
        self.inputs = tf.reshape(tf.concat([tf.reshape(placeholders['adj_fc_pre'], [FLAGS.batch_size, FLAGS.windows_size, -1]), tf.reshape(inputs, [FLAGS.batch_size, 1, -1])], 1), [FLAGS.batch_size, FLAGS.windows_size+1, self.node_num*self.node_num])
        self.adj_fc = tf.reshape(tf.concat([tf.reshape(placeholders['adj_fc_pre'], [FLAGS.batch_size, FLAGS.windows_size, -1]), tf.reshape(inputs, [FLAGS.batch_size, 1, -1])], 1), [FLAGS.batch_size, FLAGS.windows_size+1, self.node_num, self.node_num])
        self.fc_features = tf.reshape(placeholders['fc_features'][:, -1], [FLAGS.batch_size, self.node_num, -1])
        self.input_dim = self.node_num

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

        return self.predict()

    def _build(self):
        # reuse = len([t for t in tf.global_variables() if t.name.startswith(self.name)]) > 0
        # with tf.variable_scope(self.name, reuse=reuse):
        #     self.hiddeng = GraphConvolutionSparseWindows(input_dim=self.input_dim,
        #                                                  output_dim=FLAGS.hidden2,
        #                                                  adj=self.adj_fc,
        #                                                  features_nonzero=0,
        #                                                  act=tf.nn.relu,
        #                                                  batch_size=FLAGS.batch_size,
        #                                                  window_size=FLAGS.windows_size+1,
        #                                                  logging=self.logging)(tf.eye(self.node_num, batch_shape=[FLAGS.batch_size, FLAGS.windows_size+1]))
        #     self.hiddeng = tf.reshape(self.hiddeng,
        #                               [FLAGS.batch_size, FLAGS.windows_size+1, self.node_num * FLAGS.hidden2])
        #     _, self.hidden1 = InnerProductLSTM(units=self.node_num * FLAGS.hidden3)(self.hiddeng)
        #     self.hidden1 = tf.reshape(self.hidden1, [FLAGS.batch_size, -1])
        #     self.hidden2 = slim.fully_connected(self.hidden1, FLAGS.hidden5, activation_fn=tf.nn.relu)
        #     self.hidden3 = slim.fully_connected(self.hidden2, FLAGS.hidden6, activation_fn=tf.nn.relu)
        #     self.outputs = slim.fully_connected(self.hidden3, 1, activation_fn=tf.nn.sigmoid)
        self.hiddeng = GraphConvolutionSparseWindows(input_dim=self.input_dim,
                                                     output_dim=FLAGS.hidden2,
                                                     adj=self.adj_fc,
                                                     features_nonzero=0,
                                                     act=tf.nn.relu,
                                                     batch_size=FLAGS.batch_size,
                                                     window_size=FLAGS.windows_size + 1,
                                                     logging=self.logging)(
            tf.eye(self.node_num, batch_shape=[FLAGS.batch_size, FLAGS.windows_size + 1]))
        self.hiddeng = tf.reshape(self.hiddeng,
                                  [FLAGS.batch_size, FLAGS.windows_size + 1, self.node_num * FLAGS.hidden2])
        _, self.hidden1 = InnerProductLSTM(units=self.node_num * FLAGS.hidden7)(self.hiddeng)
        self.hidden1 = tf.reshape(self.hidden1, [FLAGS.batch_size, -1])
        self.hidden2 = InnerProduceDense(FLAGS.hidden5, act=tf.nn.relu, name='dense_2')(self.hidden1)
        self.hidden3 = InnerProduceDense(FLAGS.hidden6, act=tf.nn.relu, name='dense_3')(self.hidden2)
        self.outputs = InnerProduceDense(1, act=tf.nn.sigmoid, name='dense__outputs')(self.hidden3)

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
        self.labels_feature = placeholders['labels_feature']
        self.generator = generator
        # self.discriminator = discriminator
        # self.X = tf.reshape(self.labels, [FLAGS.batch_size, self.node_num*self.node_num])

        self.G_sample, self.G_feature = self.generator(placeholders, num_features, features_nonzero, num_node)
        # self.D_fake = self.discriminator(placeholders, self.G_sample, num_node, num_features)
        # self.D_real = self.discriminator(placeholders, self.X, num_node, num_features)

        # eps = tf.random_uniform([FLAGS.batch_size, 1], minval=0., maxval=1.)
        # X_inter = eps * self.X + (1. - eps) * self.G_sample  # 按照eps比例生成真假样本采样X_inter
        # grad = tf.gradients(self.discriminator(placeholders, X_inter, num_node, num_features), X_inter)[0]
        # grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
        # grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))  # 梯度惩罚项 (约束项)
        #
        # self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.D_real, labels=tf.ones_like(self.D_real)))
        # self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        # self.D_loss = self.D_loss_real + self.D_loss_fake
        # self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        preds_sub = tf.reshape(self.G_sample, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, 90*90])
        # self.pred_feature_sub = tf.reshape(self.G_feature, [FLAGS.batch_size, self.node_num*FLAGS.pred_size])
        # self.label_feature_sub = tf.reshape(self.labels_feature, [FLAGS.batch_size, self.node_num*FLAGS.pred_size])
        self.pred_feature_sub = tf.reshape(self.G_feature, [FLAGS.batch_size, FLAGS.pred_size, self.node_num])
        self.label_feature_sub = tf.reshape(self.labels_feature, [FLAGS.batch_size, FLAGS.pred_size, self.node_num])

        #pre_sub = tf.reshape(self.adj_fc[:, 7], [FLAGS.batch_size, 90*90])
        lamda = 3
        abs_lamda = 0.05
        # test_pred = 20
        # preds_sub = preds_sub[:,0:test_pred,:]
        # labels_sub = labels_sub[:,0:test_pred,:]
        # self.pred_feature_sub = self.pred_feature_sub[:, 0:test_pred,:]
        # self.label_feature_sub = self.label_feature_sub[:, 0:test_pred,:]
        self.abs_value = tf.reduce_mean(abs(preds_sub - labels_sub))
        # self.Pre_loss = tf.reduce_mean(tf.losses.huber_loss(preds_sub, labels_sub)) #+ abs_lamda*self.abs_value
        self.feature_mse = tf.reduce_mean(tf.losses.mean_squared_error(self.pred_feature_sub, self.label_feature_sub))
        self.Pre_loss = self.generator.loss+FLAGS.lamda*self.feature_mse
        self.mse = tf.reduce_mean(tf.losses.mean_squared_error(preds_sub, labels_sub))
        self.mae = tf.reduce_mean(abs(preds_sub-labels_sub))
        self.feature_mae = tf.reduce_mean(abs(self.pred_feature_sub-self.label_feature_sub))
        self.avg_preds = tf.reduce_mean(preds_sub)
        self.avg_labels = tf.reduce_mean(labels_sub)
        self.total_Pre_loss = self.Pre_loss + lamda*self.feature_mse
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        ones_like_actuals = tf.ones_like(tf.cast(tf.greater_equal(labels_sub, 0.5), tf.int32))
        zeros_like_actuals = tf.zeros_like(tf.cast(tf.greater_equal(labels_sub, 0.5), tf.int32))
        ones_like_predictions = tf.ones_like(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32))
        zeros_like_predictions = tf.zeros_like(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32))

        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.cast(tf.greater_equal(labels_sub, 0.5), tf.int32), ones_like_actuals), tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32), ones_like_predictions)), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.cast(tf.greater_equal(labels_sub, 0.5), tf.int32), zeros_like_actuals), tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32), zeros_like_predictions)), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.cast(tf.greater_equal(labels_sub, 0.5), tf.int32), zeros_like_actuals), tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32), ones_like_predictions)), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.cast(tf.greater_equal(labels_sub, 0.5), tf.int32), ones_like_actuals), tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32), zeros_like_predictions)), tf.float32))

        tpr = tp / (tp + fn)
        self.recall = tpr
        self.precision = tp / (tp + fp)
        self.accuracy = (tp+tn)/(tp+fp+fn+tn)

        # self.D_loss = tf.reduce_mean(self.D_fake - self.D_real) #+ grad_pen
        # self.G_loss = -tf.reduce_mean(self.D_fake)# + 10 * self.mse
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.pre_learning_rate = tf.train.exponential_decay(FLAGS.pre_learning_rate, global_step=self.global_step, decay_steps=10, decay_rate=0.1)
        # self.D_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0., beta2=0.9).minimize(self.D_loss, var_list=self.discriminator.vars)
        # self.G_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0., beta2=0.9).minimize(self.G_loss, var_list=self.generator.vars)
        self.Pre_solver = tf.train.AdamOptimizer(learning_rate=self.pre_learning_rate).minimize(self.Pre_loss, var_list=self.generator.vars)
        self.Pre_feature_solver = tf.train.AdamOptimizer(learning_rate=self.pre_learning_rate).minimize(self.feature_mse, var_list=self.generator.vars)
        self.Total_Pre_solver = tf.train.AdamOptimizer(learning_rate=self.pre_learning_rate).minimize(self.total_Pre_loss, var_list=self.generator.vars)


class node_LSTM(Model):
    def __init__(self, placeholders, num_features, features_nonzero, num_node, **kwargs):
        super(node_LSTM, self).__init__(**kwargs)
        self.inputs_sc = placeholders['sc_features']
        self.inputs_fc = placeholders['fc_features']
        self.input_dim = num_features
        self.node_num = num_node
        self.features_nonzero = features_nonzero
        self.adj_sc = placeholders['adj_sc']
        self.adj_fc = placeholders['adj_fc']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.labels_feature = placeholders['labels_feature']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _build(self):
        self.fc_bg = tf.reshape(self.inputs_fc[:, 0, :], [FLAGS.batch_size, self.node_num, FLAGS.window_length])
        self.fc_ed = tf.reshape(self.inputs_fc[:, FLAGS.windows_size-1, :, FLAGS.window_length-FLAGS.windows_size:], [FLAGS.batch_size, self.node_num, FLAGS.window_length-FLAGS.windows_size])
        self.inputs = tf.concat([self.fc_bg, self.fc_ed], axis=2)
        self.inputs_fc_new = tf.reshape(self.inputs, [FLAGS.batch_size, FLAGS.window_length+FLAGS.windows_size, self.node_num])
        self.lstm_fc, _ = InnerProductLSTM(units=self.node_num * FLAGS.hidden3, name='lstm_fc')(self.inputs_fc_new)
        self.pred_feature = InnerProduceDense(self.node_num*FLAGS.pred_size, act=tf.nn.sigmoid, name='gen_pred_dense')(self.lstm_fc)
        self.pred_feature = tf.reshape(self.pred_feature, [FLAGS.batch_size, self.node_num, FLAGS.pred_size])
        self.corr = t_tensor_corrcoef(tf.concat([tf.reshape(self.inputs_fc[:, -1, :, 1:],
                                                            [FLAGS.batch_size, self.node_num, FLAGS.window_length-1]),
                                                 self.pred_feature], axis=-1), FLAGS.pred_size)
        self.lstm_fc = tf.reshape(self.lstm_fc, [FLAGS.batch_size, self.node_num, FLAGS.hidden3])
        self.corr = (tf.abs(self.corr)+self.corr)/2
        self.outputs = tf.reshape(self.corr, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])
        preds_sub = tf.reshape(self.outputs, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])
        labels_sub = tf.reshape(self.labels, [FLAGS.batch_size, FLAGS.pred_size, self.node_num * self.node_num])
        test_pred = 10
        preds_sub = preds_sub[:,0:test_pred,:]
        labels_sub = labels_sub[:,0:test_pred,:]
        self.pred_feature = self.pred_feature[:, 0:test_pred,:]
        self.labels_feature = self.labels_feature[:, 0:test_pred,:]
        self.mse = tf.reduce_mean(tf.losses.mean_squared_error(preds_sub, labels_sub))
        self.feature_mse = tf.reduce_mean(tf.losses.mean_squared_error(self.pred_feature, self.labels_feature))
        self.mae = tf.reduce_mean(abs(preds_sub-labels_sub))
        self.feature_mae = tf.reduce_mean(abs(self.pred_feature-self.labels_feature))
        self.total_mse = self.mse + FLAGS.lamda*self.feature_mse
        self.solver = tf.train.AdamOptimizer(learning_rate=FLAGS.pre_learning_rate).minimize(self.mse)
        self.feature_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.pre_learning_rate).minimize(self.feature_mse)
        self.Pre_solver = tf.train.AdamOptimizer(learning_rate=FLAGS.pre_learning_rate).minimize(self.total_mse)

