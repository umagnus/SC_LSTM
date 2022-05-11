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
from models import Generator, Discriminator, Generator_new, lstmGAN, Generator_tgcn_fc, node_LSTM, Generator_fnn
from preprocessing import preprocess_graph, construct_feed_dict, construct_feed_dict_discriminator, sparse_to_tuple, mask_test_edges, getDataFortrain
from paramaters import FLAGS
from matplotlib.pyplot import savefig
from utils import test_epoch

np.set_printoptions(suppress=True)
model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
sc_features, sc_adj, fc_adj, fc_features = loadFSData("G:\Data\dMRI_preprocess\dMRI", "G:\Data\REST1")
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
generator = Generator_tgcn_fc()
discriminator = Discriminator()
lstmGAN = lstmGAN(placeholders, window_length, features_nonzero, node_number, generator, discriminator)
# lstmGAN = node_LSTM(placeholders, window_length, features_nonzero, node_number)

pos_weight = float(sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) / sc_adj[0].sum()
norm = sc_adj[0].shape[0] * sc_adj[0].shape[0] / float((sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) * 2)

# Initialize session
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=100)
pre_epoch = 0
saver.restore(sess, 'model/tgcn_fc_dense_SC_pre_much_50_lamda_1.0_gama_1.0-50')

mse = 0
feature_mse = 0
mae = 0
feature_mae = 0
for ind in index:
    sub_index=[]
    sub_index.append(ind)
    adj_sc, adj_fc, adj_label, features_sc, features_fc, adj_fc_pre, labels_feature, test_adj_sc, test_adj_fc, \
    test_adj_label, test_features_sc, test_features_fc, test_adj_fc_pre, test_labels_feature = \
        getDataFortrain(sub_index, sc_features, sc_adj, fc_adj, fc_features, fc_adj_pre, node_number)
    print("index=%r:" % ind)
    for epoch in range(FLAGS.sub_epochs):
        avg_pre_loss = 0
        avg_pre_acc = 0
        avg_pre_rec = 0
        avg_pre_pre = 0
        t = time.time()
        for i in range(len(adj_sc)):
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_sc[i], adj_fc[i], adj_label[i], features_sc[i], features_fc[i],
                                            adj_fc_pre[i], labels_feature[i], placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Run single weight update
            outs_feature = sess.run([lstmGAN.Pre_feature_solver], feed_dict=feed_dict)
            outs = sess.run(
                [lstmGAN.Pre_solver, lstmGAN.mse, lstmGAN.feature_mse, lstmGAN.total_Pre_loss, lstmGAN.precision,
                 lstmGAN.pred_feature_sub, lstmGAN.label_feature_sub], feed_dict=feed_dict)
            # Compute average loss
            avg_pre_loss = avg_pre_loss + outs[1]
            avg_pre_acc = avg_pre_acc + outs[2]
            avg_pre_rec = avg_pre_rec + outs[3]
            avg_pre_pre = avg_pre_pre + outs[4]
        sess.run([lstmGAN.add_global])
        avg_pre_loss = avg_pre_loss / len(adj_sc)
        avg_pre_acc = avg_pre_acc / len(adj_sc)
        avg_pre_rec = avg_pre_rec / len(adj_sc)
        avg_pre_pre = avg_pre_pre / len(adj_sc)
        print("PreSubEpoch:", '%04d' % (epoch + 1), "pre_train_loss=", "{:.5f}".format(avg_pre_loss),
              "pre_train_feature_mse=", "{:.5f}".format(avg_pre_acc),
              "pre_train_total_Pre_loss=", "{:.5f}".format(avg_pre_rec),
              "pre_train_precision=", "{:.5f}".format(avg_pre_pre),
              "time=", "{:.5f}".format(time.time() - t))
        if epoch % 10 == 0:
            saver.save(sess, 'model/dense_pre_%r_index=%r' % (pre_epoch, ind), global_step=epoch)
    print("sub_pre_train_index=%r" % ind + "finished!")
    test_mse, test_feature_mse, test_mae, test_feature_mae = test_epoch(sess, test_adj_sc, test_adj_fc, test_adj_label, test_features_sc, test_features_fc, test_adj_fc_pre,
               test_labels_feature, placeholders, lstmGAN, pre_epoch, ind, ax, cbar_ax)
    mse = mse+test_mse
    feature_mse = feature_mse + test_feature_mse
    mae = mae + test_mae
    feature_mae = feature_mae + test_feature_mae
print(mse, feature_mse, mae, feature_mae)
