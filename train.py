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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerED
from input_data import loadFSData
from models import Generator, Discriminator, lstmGAN, Generator_new
from preprocessing import preprocess_graph, construct_feed_dict, construct_feed_dict_discriminator, sparse_to_tuple, mask_test_edges
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
train_index = []
test_index = []
for i in range(len(sc_adj)):
    # if i < int(len(sc_adj)*0.8):
        train_index.append(i)
    # else:
        test_index.append(i)

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


def draw_graph(label_graph, pred_graph, batch, epochs, test_i, test_sub):
    fig_label = sns.heatmap(np.reshape(label_graph[batch], (90, 90)), ax=ax, cbar_ax=cbar_ax,
                            cmap='YlGnBu', vmin=0, vmax=1)
    heatmap_label = fig_label.get_figure()
    heatmap_label.savefig('./heatmap_new_2/' + 'index=' + str(test_sub) + 'iter=' + str(test_i) + '_batch=' + str(batch) +
                          'label_GAN_gcn_lstm_pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(
        epochs) + '.jpg', dpi=400)
    fig_pred = sns.heatmap(np.reshape(pred_graph[batch], (90, 90)), ax=ax, cbar_ax=cbar_ax,
                           cmap='YlGnBu', vmin=0, vmax=1)
    heatmap_pred = fig_pred.get_figure()
    heatmap_pred.savefig('./heatmap_new_2/' + 'index=' + str(test_sub) + 'iter=' + str(test_i) + '_batch=' + str(batch) +
                         'pred_GAN_gcn_lstm_pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(
        epochs) + '.jpg', dpi=400)
    del fig_label
    del fig_pred
    del heatmap_label
    del heatmap_pred
    gc.collect()

# test function
def test_epoch(test_sess, epochs):
    test_cost = 0
    test_acc = 0
    test_rec = 0
    test_pre = 0
    hm_graph = np.zeros(8100)

    for test_sub in test_index:
        test_adj_sc_sub = sc_adj[test_sub]
        test_adj_sc_sub = preprocess_graph(test_adj_sc_sub)
        test_adj_fc_sub = fc_adj[test_sub]
        test_adj_label_sub = fc_adj_pre[test_sub]
        # for ind in range(adj_fc_sub.shape[0]):
        #     adj_fc_sub[ind] = preprocess_graph(adj_fc_sub[ind])
        test_features_sc_sub = sc_features[test_sub]
        test_features_fc_sub = fc_features[test_sub]

        test_iterations = test_adj_fc_sub.shape[0]//FLAGS.batch_size - FLAGS.windows_size
        test_train_it = []
        test_test_it = []
        for test_i in range(test_iterations):
            if test_i < int(test_iterations*0.8):
                test_train_it.append(test_i)
            else:
                test_test_it.append(test_i)
        test_round_data_len = test_iterations*FLAGS.batch_size
        test_adj_fc_data = []
        test_features_fc_data = []
        test_labels_feature_data = []
        test_adj_fc_pre_data = []
        test_ydata = []
        for test_i in range(test_round_data_len):
            test_adj_fc_data.append(test_adj_fc_sub[test_i:test_i+FLAGS.windows_size])
            test_features_fc_data.append(test_features_fc_sub[test_i:test_i+FLAGS.windows_size])
            test_labels_feature_data.append(test_features_fc_sub[test_i+FLAGS.windows_size])
            test_adj_fc_pre_data.append(test_adj_label_sub[test_i:test_i+FLAGS.windows_size])
            test_ydata.append(test_adj_label_sub[test_i+FLAGS.windows_size])
        test_adj_fc_data = np.array(test_adj_fc_data).reshape([FLAGS.batch_size, test_iterations*FLAGS.windows_size, node_number, -1])
        test_features_fc_data = np.array(test_features_fc_data).reshape([FLAGS.batch_size, test_iterations*FLAGS.windows_size, node_number, -1])
        test_labels_feature_data = np.array(test_labels_feature_data).reshape([FLAGS.batch_size, test_iterations, node_number, -1])
        test_adj_fc_pre_data = np.array(test_adj_fc_pre_data).reshape([FLAGS.batch_size, test_iterations*FLAGS.windows_size, node_number, -1])
        test_ydata = np.array(test_ydata).reshape([FLAGS.batch_size, test_iterations, node_number, -1])

        # test_iterations = test_adj_fc_sub.shape[0] // (FLAGS.batch_size * FLAGS.windows_size) - 1
        # test_train_it = []
        # test_test_it = []
        # for test_i in range(test_iterations):
        #     if test_i < int(test_iterations * 0.8):
        #         test_train_it.append(test_i)
        #     else:
        #         test_test_it.append(test_i)
        # test_round_data_len = test_iterations * FLAGS.batch_size * FLAGS.windows_size
        # test_adj_fc_data = test_adj_fc_sub[:test_round_data_len].reshape(FLAGS.batch_size, test_iterations * FLAGS.windows_size,
        #                                                                  node_number, -1)
        # test_features_fc_data = test_features_fc_sub[:test_round_data_len].reshape(FLAGS.batch_size, test_iterations * FLAGS.windows_size,
        #                                                                            node_number, -1)
        # test_labels_feature_data = test_features_fc_sub[FLAGS.windows_size:test_round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size,
        #                                                                                                      test_iterations * FLAGS.windows_size,
        #                                                                                                      node_number, -1)
        # test_adj_fc_pre_data = test_adj_label_sub[:test_round_data_len].reshape(FLAGS.batch_size, test_iterations * FLAGS.windows_size,
        #                                                                         node_number, -1)
        # test_ydata = test_adj_label_sub[FLAGS.windows_size:test_round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size,
        #                                                                                                      test_iterations * FLAGS.windows_size,
        #                                                                                                      node_number, -1)
        test_cost_iter = 0
        test_acc_iter = 0
        test_rec_iter = 0
        test_pre_iter = 0
        hm_graph_iter = np.zeros(8100)
        # test_adj_fc = test_adj_fc_data[:, test_test_it[0] * FLAGS.windows_size:(test_test_it[0] + 1) * FLAGS.windows_size]
        # test_features_fc = test_features_fc_data[:, test_test_it[0] * FLAGS.windows_size:(test_test_it[0] + 1) * FLAGS.windows_size]
        time_series = []
        for test_i in test_test_it:
            test_adj_sc = test_adj_sc_sub
            test_adj_fc = test_adj_fc_data[:, test_i * FLAGS.windows_size:(test_i + 1) * FLAGS.windows_size]
            test_features_sc = test_features_sc_sub
            test_features_fc = test_features_fc_data[:, test_i * FLAGS.windows_size:(test_i + 1) * FLAGS.windows_size]
            test_labels_feature = test_labels_feature_data[:, test_i][:, :, 0]
            test_adj_fc_pre = test_adj_fc_pre_data[:, test_i * FLAGS.windows_size:(test_i + 1) * FLAGS.windows_size]
            test_adj_label = test_ydata[:, test_i]

            # Construct feed dictionary
            test_feed_dict = construct_feed_dict(test_adj_sc, test_adj_fc, test_adj_label, test_features_sc,
                                                 test_features_fc, test_adj_fc_pre, test_labels_feature, placeholders)
            # Run single weight update
            test_outs = test_sess.run([lstmGAN.mse, lstmGAN.G_sample, lstmGAN.feature_mse, lstmGAN.total_Pre_loss, lstmGAN.precision, lstmGAN.labels_feature], feed_dict=test_feed_dict)
            value_outs = test_sess.run([lstmGAN.abs_value, lstmGAN.avg_preds, lstmGAN.avg_labels], feed_dict=test_feed_dict)
            # print(value_outs[0], value_outs[1], value_outs[2])
            # print("++++++++++")
            # Compute average loss
            test_cost_iter = test_cost_iter + test_outs[0]
            test_acc_iter = test_acc_iter + test_outs[2]
            test_rec_iter = test_rec_iter + test_outs[3]
            test_pre_iter = test_pre_iter + test_outs[4]
            # print(adj_label[:, 0].reshape((FLAGS.batch_size, 8100)))
            # print(outs[1])
            # print("++++++++++++++++++++++++")
            # test_adj_fc = np.concatenate((test_adj_fc, test_outs[1].reshape((FLAGS.batch_size, FLAGS.remove_length, 90, 90))), axis=1)
            # test_adj_fc = test_adj_fc[:, 1:, :]
            # nxt_feature = np.concatenate((test_features_fc[:, -1, :, 1:], test_outs[5].reshape((FLAGS.batch_size, 90, FLAGS.remove_length))), axis=2)
            # nxt_feature = nxt_feature.reshape((FLAGS.batch_size, 1, 90, FLAGS.window_length))
            # test_features_fc = np.concatenate((test_features_fc[:, 1:, :], nxt_feature), axis=1)
            hm_graph_iter = hm_graph_iter + np.sum(abs(test_adj_label.reshape((FLAGS.batch_size, 8100)) - test_outs[1]),
                                                   axis=0) / FLAGS.batch_size
            label_graph = test_adj_label.reshape((FLAGS.batch_size, 8100))
            #
            # label_graph = np.corrcoef(test_features_fc[0, -1].reshape((node_number, FLAGS.window_length)))
            # for test_batch in range(test_features_fc.shape[0]-1):
            #     label_graph = np.concatenate((label_graph, np.corrcoef(test_features_fc[test_batch+1, -1].reshape((node_number, FLAGS.window_length)))), axis=0)
            # label_graph = label_graph.reshape((FLAGS.batch_size, 8100))
            # label_graph[label_graph < 0] = 0
            pred_graph = test_outs[1]
            # for batch in range(FLAGS.batch_size):
            draw_graph(label_graph, pred_graph, 0, epochs, test_i, test_sub)
            time_series.append(test_outs[0])

        test_cost = test_cost + test_cost_iter / len(test_test_it)
        test_acc = test_acc + test_acc_iter / len(test_test_it)
        test_rec = test_rec + test_rec_iter / len(test_test_it)
        test_pre = test_pre + test_pre_iter / len(test_test_it)
        hm_graph = hm_graph + hm_graph_iter / len(test_test_it)
        plt.figure()
        fig = plt.plot(test_test_it, time_series)
        savefig('index=' + str(test_sub))

    print("Test cost mse: " + str(test_cost / len(test_index)))
    print("Test cost feature_mse: " + str(test_acc / len(test_index)))
    print("Test cost total_Pre_loss: " + str(test_rec / len(test_index)))
    print("Test cost precision:" + str(test_pre / len(test_index)))
    f = open('test_result_0_0.5.txt', 'a')
    f.write('GAN_gcn_lstm_pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(epochs) + ' Test cost mse: ' + str(test_cost / len(test_index))
            + ' Test cost accuracy: ' + str(test_acc / len(test_index)) + ' Test cost recall: ' + str(test_rec / len(test_index))
            + ' Test cost precision: ' + str(test_pre / len(test_index)) + '\n')
    f.close()
    hm_graph = hm_graph / len(test_index)
    fig = sns.heatmap(np.reshape(hm_graph, (90, 90)), ax=ax, cbar_ax=cbar_ax, cmap='YlGnBu', vmin=0, vmax=0.15)
    heatmap = fig.get_figure()
    heatmap.savefig('./heatmap_new_0/' + 'GAN_gcn_lstm_pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(epochs) + '.jpg', dpi=400)
    # plt.savefig('./heatmap/' + 'pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(epochs) + '.jpg')
    # plt.show()

# Create model
generator = Generator_new()
discriminator = Discriminator()
lstmGAN = lstmGAN(placeholders, window_length, features_nonzero, node_number, generator, discriminator)

pos_weight = float(sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) / sc_adj[0].sum()
norm = sc_adj[0].shape[0] * sc_adj[0].shape[0] / float((sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) * 2)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#tf_vars = tf.trainable_variables()

#Pre-train model
for epoch in range(FLAGS.pre_epochs):
    avg_pre_loss = 0
    avg_pre_acc = 0
    avg_pre_rec = 0
    avg_pre_pre = 0
    t = time.time()

    for sub in train_index:
        adj_sc_sub = sc_adj[sub]
        adj_sc_sub = preprocess_graph(adj_sc_sub)
        adj_fc_sub = fc_adj[sub]
        adj_label_sub = fc_adj_pre[sub]
        #for ind in range(adj_fc_sub.shape[0]):
            #adj_fc_sub[ind] = preprocess_graph(adj_fc_sub[ind])
        features_sc_sub = sc_features[sub]
        features_fc_sub = fc_features[sub]

        iterations = adj_fc_sub.shape[0]//FLAGS.batch_size - FLAGS.windows_size
        train_it = []
        test_it = []
        for i in range(iterations):
            if i < int(iterations*0.8):
                train_it.append(i)
            else:
                test_it.append(i)
        round_data_len = iterations*FLAGS.batch_size
        adj_fc_data = []
        features_fc_data = []
        labels_feature_data = []
        adj_fc_pre_data = []
        ydata = []
        for i in range(round_data_len):
            adj_fc_data.append(adj_fc_sub[i:i+FLAGS.windows_size])
            features_fc_data.append(features_fc_sub[i:i+FLAGS.windows_size])
            labels_feature_data.append(features_fc_sub[i+FLAGS.windows_size])
            adj_fc_pre_data.append(adj_label_sub[i:i+FLAGS.windows_size])
            ydata.append(adj_label_sub[i+FLAGS.windows_size])
        adj_fc_data = np.array(adj_fc_data).reshape([FLAGS.batch_size, iterations*FLAGS.windows_size, node_number, -1])
        features_fc_data = np.array(features_fc_data).reshape([FLAGS.batch_size, iterations*FLAGS.windows_size, node_number, -1])
        labels_feature_data = np.array(labels_feature_data).reshape([FLAGS.batch_size, iterations, node_number, -1])
        adj_fc_pre_data = np.array(adj_fc_pre_data).reshape([FLAGS.batch_size, iterations*FLAGS.windows_size, node_number, -1])
        ydata = np.array(ydata).reshape([FLAGS.batch_size, iterations, node_number, -1])


        # iterations = adj_fc_sub.shape[0]//(FLAGS.batch_size*FLAGS.windows_size)-1
        # train_it = []
        # test_it = []
        # for i in range(iterations):
        #     if i < int(iterations*0.8):
        #         train_it.append(i)
        #     else:
        #         test_it.append(i)
        # round_data_len = iterations * FLAGS.batch_size * FLAGS.windows_size
        # adj_fc_data = adj_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        # features_fc_data = features_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        # labels_feature_data = features_fc_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        # adj_fc_pre_data = adj_label_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size,
        #                                                                         node_number, -1)
        # ydata = adj_label_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)

        avg_pre_loss_iter = 0
        avg_pre_acc_iter = 0
        avg_pre_rec_iter = 0
        avg_pre_pre_iter = 0
        for i in train_it:
            adj_sc = adj_sc_sub
            adj_fc = adj_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            features_sc = features_sc_sub
            features_fc = features_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            labels_feature = labels_feature_data[:, i][:, :, 0]
            adj_fc_pre = adj_fc_pre_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]
            adj_label = ydata[:, i]


            # fig0 = plt.figure(figsize=(6, 4))
            # ax = fig0.add_axes([0.2, 0.07, 0.6, 0.9], facecolor='white')
            # cbar_ax = fig0.add_axes([0.85, 0.07, 0.05, 0.88])
            # fig_pred = sns.heatmap(np.reshape(adj_label[0, 0], (90, 90)), ax=ax, cbar_ax=cbar_ax,
            #                        cmap='YlGnBu', vmin=0, vmax=1)
            # heatmap_pred = fig_pred.get_figure()
            # heatmap_pred.savefig('./heatmap_new_2/' + 'train_it=' + str(i) + '.jpg', dpi=400)

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_sc, adj_fc, adj_label, features_sc, features_fc, adj_fc_pre, labels_feature, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Run single weight update
            # outs_adj = sess.run([lstmGAN.Pre_solver], feed_dict=feed_dict)
            outs_feature = sess.run([lstmGAN.Pre_feature_solver], feed_dict=feed_dict)
            outs = sess.run([lstmGAN.Pre_solver, lstmGAN.Pre_loss, lstmGAN.feature_mse, lstmGAN.total_Pre_loss, lstmGAN.precision, lstmGAN.pred_feature_sub, lstmGAN.label_feature_sub], feed_dict=feed_dict)

            # Compute average loss
            avg_pre_loss_iter = avg_pre_loss_iter + outs[1]
            avg_pre_acc_iter = avg_pre_acc_iter + outs[2]
            avg_pre_rec_iter = avg_pre_rec_iter + outs[3]
            avg_pre_pre_iter = avg_pre_pre_iter + outs[4]
            # print(outs[4])
            # print(outs[5]-outs[6])
            # print("-----------------------")
            # print(outs[6])
            # print("+++++++++++++++++++++++")
        avg_pre_loss = avg_pre_loss + avg_pre_loss_iter/len(train_it)
        avg_pre_acc = avg_pre_acc + avg_pre_acc_iter/len(train_it)
        avg_pre_rec = avg_pre_rec + avg_pre_rec_iter/len(train_it)
        avg_pre_pre = avg_pre_pre + avg_pre_pre_iter/len(train_it)
    sess.run([lstmGAN.add_global])
    avg_pre_loss = avg_pre_loss/len(train_index)
    avg_pre_acc = avg_pre_acc/len(train_index)
    avg_pre_rec = avg_pre_rec/len(train_index)
    avg_pre_pre = avg_pre_pre/len(train_index)
    print("PreEpoch:", '%04d' % (epoch + 1), "pre_train_loss=", "{:.5f}".format(avg_pre_loss),
          "pre_train_feature_mse=", "{:.5f}".format(avg_pre_acc),
          "pre_train_total_Pre_loss=", "{:.5f}".format(avg_pre_rec),
          "pre_train_precision=", "{:.5f}".format(avg_pre_pre),
          "time=", "{:.5f}".format(time.time() - t))
print("Pre Training Finished!")

# Train model
for epoch in range(FLAGS.epochs):
    avg_d_loss = 0
    avg_g_loss = 0
    avg_mse_loss = 0
    t = time.time()
    if epoch % FLAGS.test_sp == 0:
        test_epoch(test_sess=sess, epochs=epoch)

    for sub in train_index:
        adj_sc_sub = sc_adj[sub]
        adj_sc_sub = preprocess_graph(adj_sc_sub)
        adj_fc_sub = fc_adj[sub]
        adj_label_sub = fc_adj_pre[sub]
        #for ind in range(adj_fc_sub.shape[0]):
            #adj_fc_sub[ind] = preprocess_graph(adj_fc_sub[ind])
        features_sc_sub = sc_features[sub]
        features_fc_sub = fc_features[sub]

        iterations = adj_fc_sub.shape[0]//(FLAGS.batch_size*FLAGS.windows_size)-1
        train_it = []
        test_it = []
        for i in range(iterations):
            if i < int(iterations*0.8):
                train_it.append(i)
            else:
                test_it.append(i)
        round_data_len = iterations * FLAGS.batch_size * FLAGS.windows_size
        adj_fc_data = adj_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        features_fc_data = features_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        labels_feature_data = features_fc_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        adj_fc_pre_data = adj_label_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size,
                                                                                node_number, -1)
        ydata = adj_label_sub[FLAGS.windows_size:round_data_len + FLAGS.windows_size].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)

        avg_d_loss_iter = 0
        avg_g_loss_iter = 0
        avg_mse_loss_iter = 0
        for i in train_it:
            adj_sc = adj_sc_sub
            adj_fc = adj_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            features_sc = features_sc_sub
            features_fc = features_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            labels_feature = labels_feature_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size][:, 0, :, 0]
            adj_fc_pre = adj_fc_pre_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]
            adj_label = ydata[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_sc, adj_fc, adj_label[:, 0], features_sc, features_fc, adj_fc_pre, labels_feature, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            #feed_dict.update({real_graph: np.reshape(adj_label[:, 0], [FLAGS.batch_size, -1])})

            #D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
            #d_vars = [var for var in tf_vars if var.name.startswith('discriminator')]
            #D_opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(D_loss, var_list=d_vars)

            # Run single weight update
            #_, D_loss_curr = sess.run([D_opt, D_loss], feed_dict=feed_dict)
            #outs = sess.run([model.opt_op, model.loss, model.outputs, model.labels], feed_dict=feed_dict)
            outs_d = sess.run([lstmGAN.D_solver, lstmGAN.D_loss], feed_dict=feed_dict)
            outs_g = sess.run([lstmGAN.G_solver, lstmGAN.G_loss, lstmGAN.mse, lstmGAN.label_feature_sub, lstmGAN.pred_feature_sub], feed_dict=feed_dict)


            # Compute average loss
            avg_d_loss_iter = avg_d_loss_iter + outs_d[1]
            avg_g_loss_iter = avg_g_loss_iter + outs_g[1]
            avg_mse_loss_iter = avg_mse_loss_iter + outs_g[2]
            # print(outs_g[3]-outs_g[4])
            # print('---------------')
            # print(outs_g[4])
            # print('++++++++++++++++')
            # print(outs_g[3])
            # print('******************')
        avg_d_loss = avg_d_loss + avg_d_loss_iter/iterations
        avg_g_loss = avg_g_loss + avg_g_loss_iter/iterations
        avg_mse_loss = avg_mse_loss + avg_mse_loss_iter/iterations
    avg_d_loss = avg_d_loss/len(train_index)
    avg_g_loss = avg_g_loss/len(train_index)
    avg_mse_loss = avg_mse_loss/len(train_index)
    print("Epoch:", '%04d' % (epoch + 1), "train_d_loss=", "{:.5f}".format(avg_d_loss),
          "train_g_loss=", "{:.5f}".format(avg_g_loss),
          "train_mse_loss=", "{:.5f}".format(avg_mse_loss),
          "time=", "{:.5f}".format(time.time() - t))


print("Optimization Finished!")

# roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))
