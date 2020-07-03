from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerED
from input_data import loadFSData
from models import Generator, Discriminator
from preprocessing import preprocess_graph, construct_feed_dict, construct_feed_dict_discriminator, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.(sc)')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.(fc)')
flags.DEFINE_integer('hidden3', 256, 'Number of units in hidden layer 3.(lstm)')
flags.DEFINE_integer('hidden4', 128, 'Number of units in hidden layer 4.(dis)')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('windows_size', 1, 'Length of windows')
flags.DEFINE_integer('batch_size', 16, 'Batch size')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
sc_features, sc_adj, fc_adj, fc_features = loadFSData("F:\Data\dMRI", "F:\Data\REST1")
fc_adj_pre = fc_adj
for i in range(len(fc_adj)):
    for j in range(len(fc_adj[i])):
        fc_adj_pre[i][j] = preprocess_graph(fc_adj[i][j])
# fc_adj = fc_adj_pre
train_index = []
test_index = []
for i in range(len(sc_adj)):
    if i <= int(len(sc_adj)*0.8):
        train_index.append(i)
    else:
        test_index.append(i)


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])


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
model = Generator(placeholders, window_length, features_nonzero, node_number)
#real_graph = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, node_number*node_number], name='real_graph')
#D_fake = Discriminator(model.pred)
#D_real = Discriminator(real_graph)

pos_weight = float(sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) / sc_adj[0].sum()
norm = sc_adj[0].shape[0] * sc_adj[0].shape[0] / float((sc_adj[0].shape[0] * sc_adj[0].shape[0] - sc_adj[0].sum()) * 2)


# Optimizer
with tf.name_scope('optimizer'):
    opt = OptimizerED(preds=model.pred,
                      labels=placeholders['labels'],
                      pos_weight=pos_weight,
                      norm=10000)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#tf_vars = tf.trainable_variables()

# Train model
for epoch in range(FLAGS.epochs):
    avg_cost = 0
    avg_accuracy = 0
    t = time.time()

    for sub in train_index:
        adj_sc_sub = sc_adj[sub]
        adj_sc_sub = preprocess_graph(adj_sc_sub)
        adj_fc_sub = fc_adj_pre[sub]
        adj_label_sub = fc_adj[sub]
        #for ind in range(adj_fc_sub.shape[0]):
            #adj_fc_sub[ind] = preprocess_graph(adj_fc_sub[ind])
        features_sc_sub = sc_features[sub]
        features_fc_sub = fc_features[sub]

        iterations = adj_fc_sub.shape[0]//(FLAGS.batch_size*FLAGS.windows_size)
        round_data_len = iterations * FLAGS.batch_size * FLAGS.windows_size
        adj_fc_data = adj_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        features_fc_data = features_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
        ydata = adj_label_sub[1:round_data_len + 1].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)

        avg_cost_iter = 0
        avg_accuracy_iter = 0
        for i in range(iterations):
            adj_sc = adj_sc_sub
            adj_fc = adj_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            features_sc = features_sc_sub
            features_fc = features_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]
            adj_label = ydata[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size]

            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_sc, adj_fc, adj_label[:, 0], features_sc, features_fc, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            #feed_dict.update({real_graph: np.reshape(adj_label[:, 0], [FLAGS.batch_size, -1])})

            #D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
            #d_vars = [var for var in tf_vars if var.name.startswith('discriminator')]
            #D_opt = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(D_loss, var_list=d_vars)

            # Run single weight update
            #_, D_loss_curr = sess.run([D_opt, D_loss], feed_dict=feed_dict)
            outs = sess.run([opt.opt_op, opt.cost, opt.mse, opt.pred, opt.label], feed_dict=feed_dict)

            # Compute average loss
            avg_cost_iter = avg_cost_iter + outs[1]
            avg_accuracy_iter = avg_accuracy_iter + outs[2]
            # print(outs[4])
            # print(outs[3])
            # print(outs[2])
            # print("+++++++++++++++++++++++")
        avg_cost = avg_cost + avg_cost_iter/iterations
        avg_accuracy = avg_accuracy + avg_accuracy_iter/iterations
    avg_cost = avg_cost/len(train_index)
    avg_accuracy = avg_accuracy/len(train_index)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_mse=", "{:.5f}".format(avg_accuracy),
       #"D_loss=", "{:.5f}".format(D_loss_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

test_cost = 0
test_accuracy = 0
file = open('test_result.txt', 'w')
for sub in test_index:
    adj_sc_sub = sc_adj[sub]
    adj_sc_sub = preprocess_graph(adj_sc_sub)
    adj_fc_sub = fc_adj_pre[sub]
    adj_label_sub = fc_adj[sub]
    # for ind in range(adj_fc_sub.shape[0]):
    #     adj_fc_sub[ind] = preprocess_graph(adj_fc_sub[ind])
    features_sc_sub = sc_features[sub]
    features_fc_sub = fc_features[sub]

    iterations = adj_fc_sub.shape[0] // (FLAGS.batch_size * FLAGS.windows_size)
    round_data_len = iterations * FLAGS.batch_size * FLAGS.windows_size
    adj_fc_data = adj_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number,
                                                      -1)
    features_fc_data = features_fc_sub[:round_data_len].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size,
                                                                node_number, -1)
    ydata = adj_label_sub[1:round_data_len + 1].reshape(FLAGS.batch_size, iterations * FLAGS.windows_size, node_number, -1)
    test_cost_iter = 0
    test_accuracy_iter = 0
    for i in range(iterations):
        adj_sc = adj_sc_sub
        adj_fc = adj_fc_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]
        features_sc = features_sc_sub
        features_fc = features_fc_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]
        adj_label = ydata[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size]

        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_sc, adj_fc, adj_label[:, 0], features_sc, features_fc, placeholders)
        # Run single weight update
        outs = sess.run([opt.mse, opt.accuracy, model.pred], feed_dict=feed_dict)

        # Compute average loss
        test_cost_iter = test_cost_iter + outs[0]
        test_accuracy_iter = test_accuracy_iter + outs[1]
        print(adj_label[:, 0].reshape((FLAGS.batch_size, 8100)))
        print(outs[2])
        print("++++++++++++++++++++++++")

    test_cost = test_cost+test_cost_iter/iterations
    test_accuracy = test_accuracy + test_accuracy_iter/iterations

print("Test cost mse: " + str(test_cost/len(test_index)))
print("Test accuracy: " + str(test_accuracy/len(test_index)))
file.close()
# roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))