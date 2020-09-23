from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import copy
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerED
from input_data import loadFSData
from models import Generator, Discriminator, lstmGAN
from preprocessing import preprocess_graph, construct_feed_dict, construct_feed_dict_discriminator, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('pre_learning_rate', 0.0001, 'Initial pre learning rate')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('pre_epochs', 500, 'Number of epochs to pre-train')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.(sc)')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.(fc)')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 3.(lstm)')
flags.DEFINE_integer('hidden4', 8, 'Number of units in hidden layer 4.(dis)')
flags.DEFINE_integer('hidden5', 128, 'Number of units in hidden layer 5.(dis_fc)')
flags.DEFINE_integer('hidden6', 16, 'Number of units in hidden layer 6.(dis_fc)')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('windows_size', 8, 'Length of windows')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('cp_step', 50, 'checkpoint_step')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
sc_features, sc_adj, fc_adj, fc_features = loadFSData("H:\Data\dMRI", "H:\Data\REST1")
fc_adj_pre = copy.deepcopy(fc_adj)
for i in range(len(fc_adj)):
    for j in range(len(fc_adj[i])):
        fc_adj[i][j] = preprocess_graph(fc_adj[i][j])
# fc_adj = fc_adj_pre
train_index = []
test_index = []
for i in range(len(sc_adj)):
    if i <= int(len(sc_adj)*0.8):
        train_index.append(i)
    else:
        test_index.append(i)

# Define placeholders
placeholders = {
    'sc_features': tf.placeholder(tf.float32),
    'fc_features': tf.placeholder(tf.float32),
    'adj_fc': tf.placeholder(tf.float32),
    'adj_sc': tf.placeholder(tf.float32),
    'labels': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

time_length = 1200
window_length = 40
node_number = 90
remove_length = 1
window_number = time_length - window_length
num_features = 90
features_nonzero = 890
# Create model
generator = Generator()
discriminator = Discriminator()
lstmGAN = lstmGAN(placeholders, window_length, features_nonzero, node_number, generator, discriminator)

pos_weight = float(sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) / sc_adj[0].sum()
norm = sc_adj[0].shape[0] * sc_adj[0].shape[0] / float((sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) * 2)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=50)
saver_path = 'model/'
#tf_vars = tf.trainable_variables()

#Pre-train model
for epoch in range(FLAGS.pre_epochs):
    avg_pre_loss = 0
    t = time.time()
    if epoch % FLAGS.cp_step == 0:
        saver.save(sess=sess, save_path=saver_path + 'pre_train_' + str(epoch) + '.cpkt')
        print("Save model PreEpoch", '%04d' % epoch)

    for sub in train_index:
        adj_sc_sub = sc_adj[sub]
        adj_sc_sub = preprocess_graph(adj_sc_sub)
        adj_fc_sub = fc_adj[sub]
        adj_label_sub = fc_adj_pre[sub]
        features_sc_sub = sc_features[sub]
        features_fc_sub = fc_features[sub]

        iterations = adj_fc_sub.shape[0]//(FLAGS.batch_size*FLAGS.windows_size)-1
        round_data_len = iterations * FLAGS.batch_size * FLAGS.windows_size
        adj_fc_data = adj_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        features_fc_data = features_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        ydata = adj_label_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)

        avg_pre_loss_iter = 0
        for i in range(iterations):
            adj_sc = adj_sc_sub
            adj_fc = adj_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            features_sc = features_sc_sub
            features_fc = features_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            adj_label = ydata[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_sc, adj_fc, adj_label[:, 0], features_sc, features_fc, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Run single weight update
            outs = sess.run([lstmGAN.Pre_solver, lstmGAN.Pre_loss], feed_dict=feed_dict)

            # Compute average loss
            avg_pre_loss_iter = avg_pre_loss_iter + outs[1]
            # print(outs[4])
            # print(outs[3])
            # print(outs[2])
            # print("+++++++++++++++++++++++")
        avg_pre_loss = avg_pre_loss + avg_pre_loss_iter/iterations
    avg_pre_loss = avg_pre_loss/len(train_index)
    print("PreEpoch:", '%04d' % (epoch + 1), "pre_train_loss=", "{:.5f}".format(avg_pre_loss),
          "time=", "{:.5f}".format(time.time() - t))

print("Pre Training Finished!")
