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
flags.DEFINE_integer('pre_epochs', 100, 'Number of epochs to pre-train')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
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
saver_path = 'model/'
# new_saver = tf.train.import_meta_graph(saver_path + 'pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(FLAGS.epochs)
#                                        + '.cpkt-' + str(FLAGS.epochs) + '.meta')
# new_saver.restore(sess, saver_path + 'pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(FLAGS.epochs) + '.cpkt-' +
#                   str(FLAGS.epochs))
new_saver = tf.train.import_meta_graph(saver_path + 'pre_train_' + str(FLAGS.pre_epochs) + '.cpkt.meta')
new_saver.restore(sess, saver_path + 'pre_train_' + str(FLAGS.pre_epochs) + '.cpkt')


test_cost = 0
# test_accuracy = 0
hm_graph = np.zeros(8100)
for sub in test_index:
    adj_sc_sub = sc_adj[sub]
    adj_sc_sub = preprocess_graph(adj_sc_sub)
    adj_fc_sub = fc_adj[sub]
    adj_label_sub = fc_adj_pre[sub]
    # for ind in range(adj_fc_sub.shape[0]):
    #     adj_fc_sub[ind] = preprocess_graph(adj_fc_sub[ind])
    features_sc_sub = sc_features[sub]
    features_fc_sub = fc_features[sub]

    iterations = adj_fc_sub.shape[0] // (FLAGS.batch_size * FLAGS.windows_size)-1
    round_data_len = iterations * FLAGS.batch_size * FLAGS.windows_size
    adj_fc_data = adj_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number,
                                                      -1)
    features_fc_data = features_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size,
                                                                node_number, -1)
    ydata = adj_label_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
    test_cost_iter = 0
    # test_accuracy_iter = 0
    hm_graph_iter = np.zeros(8100)
    for i in range(iterations):
        adj_sc = adj_sc_sub
        adj_fc = adj_fc_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]
        features_sc = features_sc_sub
        features_fc = features_fc_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]
        adj_label = ydata[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]

        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_sc, adj_fc, adj_label[:, 0], features_sc, features_fc, placeholders)
        # Run single weight update
        outs = sess.run([lstmGAN.mse, lstmGAN.G_sample], feed_dict=feed_dict)

        # Compute average loss
        test_cost_iter = test_cost_iter + outs[0]
        # test_accuracy_iter = test_accuracy_iter + outs[1]
        # print(adj_label[:, 0].reshape((FLAGS.batch_size, 8100)))
        # print(outs[1])
        # print("++++++++++++++++++++++++")
        hm_graph_iter = hm_graph_iter+np.sum(abs(adj_label[:, 0].reshape((FLAGS.batch_size, 8100))-outs[1]), axis=0)/FLAGS.batch_size
        # heatmap = sns.heatmap(abs(adj_label[:, 0].reshape((FLAGS.batch_size, 8100))-outs[1]), cmap='YlGnBu')
        # plt.show()

    test_cost = test_cost+test_cost_iter/iterations
    # test_accuracy = test_accuracy + test_accuracy_iter/iterations
    hm_graph = hm_graph + hm_graph_iter/iterations

print("Test cost mse: " + str(test_cost/len(test_index)))
# print("Test accuracy: " + str(test_accuracy/len(test_index)))
hm_graph = hm_graph/len(test_index)
heatmap = sns.heatmap(np.reshape(hm_graph, (90, 90)), cmap='YlGnBu', vmin=0, vmax=0.5)
plt.show()
# roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))
