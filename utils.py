import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import seaborn as sns
from paramaters import FLAGS
import gc
from preprocessing import construct_feed_dict
import tensorflow as tf


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


# def construct_feed_dict(features, support, labels, labels_mask, placeholders):
#     """Construct feed dictionary."""
#     feed_dict = dict()
#     feed_dict.update({placeholders['labels']: labels})
#     feed_dict.update({placeholders['labels_mask']: labels_mask})
#     feed_dict.update({placeholders['features']: features})
#     feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
#     feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
#     return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def draw_graph(label_graph, pred_graph, batch, epochs, test_i, test_sub, ax, cbar_ax):
    fig_label = sns.heatmap(np.reshape(label_graph[batch], (90, 90)), ax=ax, cbar_ax=cbar_ax,
                            cmap='YlGnBu', vmin=0, vmax=1)
    heatmap_label = fig_label.get_figure()
    heatmap_label.savefig('./heatmap_new_3/' + 'index=' + str(test_sub) + 'iter=' + str(test_i) + '_batch=' + str(batch) +
                          'label_GAN_gcn_lstm_pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(
        epochs) + '.jpg', dpi=400)
    fig_pred = sns.heatmap(np.reshape(pred_graph[batch], (90, 90)), ax=ax, cbar_ax=cbar_ax,
                           cmap='YlGnBu', vmin=0, vmax=1)
    heatmap_pred = fig_pred.get_figure()
    heatmap_pred.savefig('./heatmap_new_3/' + 'index=' + str(test_sub) + 'iter=' + str(test_i) + '_batch=' + str(batch) +
                         'pred_GAN_gcn_lstm_pre_train_' + str(FLAGS.pre_epochs) + '_train_' + str(
        epochs) + '.jpg', dpi=400)
    del fig_label
    del fig_pred
    del heatmap_label
    del heatmap_pred
    gc.collect()


# test function
def test_epoch(test_sess, test_adj_sc, test_adj_fc, test_adj_label, test_features_sc, test_features_fc, test_adj_fc_pre,
               test_labels_feature, placeholders, model, epoch, index, ax, cbar_ax):
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
        outs = test_sess.run(
            [model.mse, model.G_sample, model.feature_mse, model.total_Pre_loss, model.precision,
             model.labels_feature], feed_dict=feed_dict)
        value_outs = test_sess.run([model.pred_feature_sub, model.label_feature_sub], feed_dict=feed_dict)
        test_mse = test_mse + outs[0]
        test_feature_mse = test_feature_mse + outs[2]
        test_total_pre_loss = test_total_pre_loss + outs[3]
        label_graph = test_adj_label[i].reshape((FLAGS.batch_size, 8100))
        pred_graph = outs[1]
        draw_graph(label_graph, pred_graph, 0, epoch, i, index, ax, cbar_ax)
    test_mse = test_mse/len(test_adj_sc)
    test_feature_mse = test_feature_mse/len(test_adj_sc)
    test_total_pre_loss = test_total_pre_loss/len(test_adj_sc)
    print("Test cost mse: " + str(test_mse))
    print("Test cost feature_mse: " + str(test_feature_mse))
    print("Test cost total_Pre_loss: " + str(test_total_pre_loss))
    f = open('test_result_pre.txt', 'a')
    f.write('GAN_gcn_lstm_pre_train_' + str(epoch) + ' Test cost mse: ' + str(test_mse)
            + ' Test cost feature mse: ' + str(test_feature_mse) + ' Test total cost: ' + str(test_total_pre_loss)
            + '\n')
    f.close()


def tensor_corrcoef(tensor):
    X = tensor
    X_T = tf.transpose(tensor, [0, 2, 1])
    matrix_x_mut_y = tf.matmul(X, X_T)*FLAGS.window_length
    matrix_x_sum = tf.reshape(tf.reduce_sum(X, axis=[2]), [FLAGS.batch_size, 90, 1])
    matrix_x_sum_T = tf.transpose(matrix_x_sum, [0, 2, 1])
    matrix_x_sum_y = tf.matmul(matrix_x_sum, matrix_x_sum_T)
    matrix_x_squ_sum = tf.reshape(tf.reduce_sum(tf.square(X), axis=[2]), [FLAGS.batch_size, 90, 1])
    matrix_x_sum_squ = tf.square(matrix_x_sum)
    matrix_sqr = tf.sqrt(FLAGS.window_length*matrix_x_squ_sum-matrix_x_sum_squ)
    matrix_sqr_reshape = tf.reshape(matrix_sqr, [FLAGS.batch_size, 1, 90])
    corr = (matrix_x_mut_y-matrix_x_sum_y)/(tf.matmul(tf.transpose(matrix_sqr_reshape, [0, 2, 1]), matrix_sqr_reshape))
    return corr

