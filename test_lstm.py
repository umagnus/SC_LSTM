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
import gc
import random

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerED
from input_data import loadFSData
from models import node_LSTM
from preprocessing import preprocess_graph, construct_feed_dict, construct_feed_dict_discriminator, sparse_to_tuple, mask_test_edges, getDataFortrain
from paramaters import FLAGS
from matplotlib.pyplot import savefig
from utils import test_epoch

np.set_printoptions(suppress=True)
model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
sc_features, sc_adj, fc_adj, fc_features = loadFSData("G:\Data\SC_LSTM test\dMRI", "G:\Data\SC_LSTM test\REST1")
fc_adj_pre = copy.deepcopy(fc_adj)
for i in range(len(fc_adj)):
    for j in range(len(fc_adj[i])):
        fc_adj[i][j] = preprocess_graph(fc_adj[i][j])
# fc_adj = fc_adj_pre
index = []
for i in range(len(sc_adj)):
    index.append(i)

# Define placeholders
placeholders = {
    'sc_features': tf.placeholder(tf.float32),
    'fc_features': tf.placeholder(tf.float32),
    'adj_fc': tf.placeholder(tf.float32),
    'adj_sc': tf.placeholder(tf.float32),
    'labels': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'adj_fc_pre': tf.placeholder(tf.float32),
    'labels_feature': tf.placeholder(tf.float32)
}

time_length = 1200
window_length = FLAGS.window_length
node_number = 90
remove_length = 1
window_number = time_length - window_length
num_features = 90
features_nonzero = 890

# figure ax
fig0 = plt.figure(figsize=(6, 4))
ax = fig0.add_axes([0.2, 0.07, 0.6, 0.9], facecolor='white')
cbar_ax = fig0.add_axes([0.85, 0.07, 0.05, 0.88])



# Create model
node_LSTM = node_LSTM(placeholders, window_length, features_nonzero, node_number)

pos_weight = float(sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) / sc_adj[0].sum()
norm = sc_adj[0].shape[0] * sc_adj[0].shape[0] / float((sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) * 2)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=100)
pre_epoch = 0
# saver.restore(sess, 'model/lstm_pre_20-20')

for ind in index:
    sub_index=[]
    sub_index.append(ind)
    adj_sc, adj_fc, adj_label, features_sc, features_fc, adj_fc_pre, labels_feature, test_adj_sc, test_adj_fc, \
    test_adj_label, test_features_sc, test_features_fc, test_adj_fc_pre, test_labels_feature = \
        getDataFortrain(sub_index, sc_features, sc_adj, fc_adj, fc_features, fc_adj_pre, node_number)
    print("index=%r:" % ind)
    test_mse = 0
    test_feature_mse = 0
    test_total_pre_loss = 0
    for i in range(len(test_adj_sc)):
        # Construct feed dictionary
        feed_dict = construct_feed_dict(test_adj_sc[i], test_adj_fc[i], test_adj_label[i], test_features_sc[i],
                                             test_features_fc[i], test_adj_fc_pre[i], test_labels_feature[i],
                                             placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run(
            [node_LSTM.mse, node_LSTM.feature_mse], feed_dict=feed_dict)
        test_mse = test_mse + outs[0]
        test_feature_mse = test_feature_mse + outs[1]
    test_mse = test_mse/len(test_adj_sc)
    test_feature_mse = test_feature_mse/len(test_adj_sc)
    test_total_pre_loss = test_total_pre_loss/len(test_adj_sc)
    print("Test cost mse: " + str(test_mse))
    print("Test cost feature_mse: " + str(test_feature_mse))
    print("Test cost total_Pre_loss: " + str(test_total_pre_loss))
