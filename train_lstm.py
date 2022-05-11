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


adj_sc, adj_fc, adj_label, features_sc, features_fc, adj_fc_pre, labels_feature, test_adj_sc, test_adj_fc, test_adj_label, test_features_sc, test_features_fc, test_adj_fc_pre, test_labels_feature = getDataFortrain(index, sc_features, sc_adj, fc_adj, fc_features, fc_adj_pre, node_number)

# Create model
node_LSTM = node_LSTM(placeholders, window_length, features_nonzero, node_number)


pos_weight = float(sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) / sc_adj[0].sum()
norm = sc_adj[0].shape[0] * sc_adj[0].shape[0] / float((sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) * 2)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# saver.restore(sess, 'model/pre_20-20')

#Pre-train model
for epoch in range(FLAGS.pre_epochs):
    avg_pre_loss = 0
    avg_pre_acc = 0
    avg_pre_rec = 0
    avg_pre_pre = 0
    t = time.time()
    #
    # for sub in index:
    #     adj_sc_sub = sc_adj[sub]
    #     adj_sc_sub = preprocess_graph(adj_sc_sub)
    #     adj_fc_sub = fc_adj[sub]
    #     adj_label_sub = fc_adj_pre[sub]
    #     #for ind in range(adj_fc_sub.shape[0]):
    #         #adj_fc_sub[ind] = preprocess_graph(adj_fc_sub[ind])
    #     features_sc_sub = sc_features[sub]
    #     features_fc_sub = fc_features[sub]
    #
    #     iterations = adj_fc_sub.shape[0]//FLAGS.batch_size - FLAGS.windows_size
    #     train_it = []
    #     test_it = []
    #     for i in range(iterations):
    #         if i < int(iterations*0.8):
    #             train_it.append(i)
    #         else:
    #             test_it.append(i)
    #     round_data_len = iterations*FLAGS.batch_size
    #     adj_fc_data = []
    #     features_fc_data = []
    #     labels_feature_data = []
    #     adj_fc_pre_data = []
    #     ydata = []
    #     for i in range(round_data_len):
    #         adj_fc_data.append(adj_fc_sub[i:i+FLAGS.windows_size])
    #         features_fc_data.append(features_fc_sub[i:i+FLAGS.windows_size])
    #         labels_feature_data.append(features_fc_sub[i+FLAGS.windows_size])
    #         adj_fc_pre_data.append(adj_label_sub[i:i+FLAGS.windows_size])
    #         ydata.append(adj_label_sub[i+FLAGS.windows_size])
    #     adj_fc_data = np.array(adj_fc_data).reshape([FLAGS.batch_size, iterations*FLAGS.windows_size, node_number, -1])
    #     features_fc_data = np.array(features_fc_data).reshape([FLAGS.batch_size, iterations*FLAGS.windows_size, node_number, -1])
    #     labels_feature_data = np.array(labels_feature_data).reshape([FLAGS.batch_size, iterations, node_number, -1])
    #     adj_fc_pre_data = np.array(adj_fc_pre_data).reshape([FLAGS.batch_size, iterations*FLAGS.windows_size, node_number, -1])
    #     ydata = np.array(ydata).reshape([FLAGS.batch_size, iterations, node_number, -1])
    #
    #
    #     # iterations = adj_fc_sub.shape[0]//(FLAGS.batch_size*FLAGS.windows_size)-1
    #     # train_it = []
    #     # test_it = []
    #     # for i in range(iterations):
    #     #     if i < int(iterations*0.8):
    #     #         train_it.append(i)
    #     #     else:
    #     #         test_it.append(i)
    #     # round_data_len = iterations * FLAGS.batch_size * FLAGS.windows_size
    #     # adj_fc_data = adj_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
    #     # features_fc_data = features_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
    #     # labels_feature_data = features_fc_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
    #     # adj_fc_pre_data = adj_label_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size,
    #     #                                                                         node_number, -1)
    #     # ydata = adj_label_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
    #
    #     avg_pre_loss_iter = 0
    #     avg_pre_acc_iter = 0
    #     avg_pre_rec_iter = 0
    #     avg_pre_pre_iter = 0
    #     for i in train_it:
    #         adj_sc = adj_sc_sub
    #         adj_fc = adj_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
    #         features_sc = features_sc_sub
    #         features_fc = features_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
    #         labels_feature = labels_feature_data[:, i][:, :, 0]
    #         adj_fc_pre = adj_fc_pre_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]
    #         adj_label = ydata[:, i]
    #
    #
    #         # fig0 = plt.figure(figsize=(6, 4))
    #         # ax = fig0.add_axes([0.2, 0.07, 0.6, 0.9], facecolor='white')
    #         # cbar_ax = fig0.add_axes([0.85, 0.07, 0.05, 0.88])
    #         # fig_pred = sns.heatmap(np.reshape(adj_label[0, 0], (90, 90)), ax=ax, cbar_ax=cbar_ax,
    #         #                        cmap='YlGnBu', vmin=0, vmax=1)
    #         # heatmap_pred = fig_pred.get_figure()
    #         # heatmap_pred.savefig('./heatmap_new_2/' + 'train_it=' + str(i) + '.jpg', dpi=400)
    #
    #         # Construct feed dictionary
    #         feed_dict = construct_feed_dict(adj_sc, adj_fc, adj_label, features_sc, features_fc, adj_fc_pre, labels_feature, placeholders)
    #         feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    #
    #         # Run single weight update
    #         # outs_adj = sess.run([lstmGAN.Pre_solver], feed_dict=feed_dict)
    #         outs_feature = sess.run([lstmGAN.Pre_feature_solver], feed_dict=feed_dict)
    #         outs = sess.run([lstmGAN.Pre_solver, lstmGAN.Pre_loss, lstmGAN.feature_mse, lstmGAN.total_Pre_loss, lstmGAN.precision, lstmGAN.pred_feature_sub, lstmGAN.label_feature_sub], feed_dict=feed_dict)
    #
    #         # Compute average loss
    #         avg_pre_loss_iter = avg_pre_loss_iter + outs[1]
    #         avg_pre_acc_iter = avg_pre_acc_iter + outs[2]
    #         avg_pre_rec_iter = avg_pre_rec_iter + outs[3]
    #         avg_pre_pre_iter = avg_pre_pre_iter + outs[4]
    #         # print(outs[4])
    #         # print(outs[5]-outs[6])
    #         # print("-----------------------")
    #         # print(outs[6])
    #         # print("+++++++++++++++++++++++")
    #     avg_pre_loss = avg_pre_loss + avg_pre_loss_iter/len(train_it)
    #     avg_pre_acc = avg_pre_acc + avg_pre_acc_iter/len(train_it)
    #     avg_pre_rec = avg_pre_rec + avg_pre_rec_iter/len(train_it)
    #     avg_pre_pre = avg_pre_pre + avg_pre_pre_iter/len(train_it)
    ind = list(range(len(adj_sc)))
    random.shuffle(ind)
    for i in ind:
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_sc[i], adj_fc[i], adj_label[i], features_sc[i], features_fc[i],
                                        adj_fc_pre[i], labels_feature[i], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Run single weight update
        outs = sess.run(
            [node_LSTM.feature_solver, node_LSTM.solver, node_LSTM.mse, node_LSTM.feature_mse], feed_dict=feed_dict)
        # Compute average loss
        avg_pre_loss = avg_pre_loss + outs[2]
        avg_pre_acc = avg_pre_acc + outs[3]
    avg_pre_loss = avg_pre_loss/len(adj_sc)
    avg_pre_acc = avg_pre_acc/len(adj_sc)
    print("PreEpoch:", '%04d' % (epoch + 1), "pre_train_loss=", "{:.5f}".format(avg_pre_loss),
          "pre_train_feature_mse=", "{:.5f}".format(avg_pre_acc),
          "time=", "{:.5f}".format(time.time() - t))
    if epoch % 10 == 0:
        saver.save(sess, 'model/lstm_pre_%r' % epoch, global_step=epoch)
print("Pre Training Finished!")
