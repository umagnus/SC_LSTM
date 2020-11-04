import numpy as np
import scipy.sparse as sp
from paramaters import FLAGS


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.todense()


def construct_feed_dict(adj_sc, adj_fc, labels, sc_features, fc_features, adj_fc_pre, labels_feature, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['sc_features']: sc_features})
    feed_dict.update({placeholders['fc_features']: fc_features})
    feed_dict.update({placeholders['adj_sc']: adj_sc})
    feed_dict.update({placeholders['adj_fc']: adj_fc})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['adj_fc_pre']: adj_fc_pre})
    feed_dict.update({placeholders['labels_feature']: labels_feature})
    return feed_dict


def construct_feed_dict_discriminator(model_pred, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['model_pred']: model_pred})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def getDataFortrain(index, sc_features, sc_adj, fc_adj, fc_features, fc_adj_pre, node_number):
    adj_sc, adj_fc, adj_label, features_sc, features_fc, adj_fc_pre, labels_feature = [], [], [], [], [], [], []
    test_adj_sc, test_adj_fc, test_adj_label, test_features_sc, test_features_fc, test_adj_fc_pre, test_labels_feature = [], [], [], [], [], [], []
    for sub in index:
        adj_sc_sub = sc_adj[sub]
        adj_sc_sub = preprocess_graph(adj_sc_sub)
        adj_fc_sub = fc_adj[sub]
        adj_label_sub = fc_adj_pre[sub]
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

        for i in train_it:
            adj_sc.append(adj_sc_sub)
            adj_fc.append(adj_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size])
            features_sc.append(features_sc_sub)
            features_fc.append(features_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size])
            labels_feature.append(labels_feature_data[:, i][:, :, 0])
            adj_fc_pre.append(adj_fc_pre_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size])
            adj_label.append(ydata[:, i])
        for i in test_it:
            test_adj_sc.append(adj_sc_sub)
            test_adj_fc.append(adj_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size])
            test_features_sc.append(features_sc_sub)
            test_features_fc.append(features_fc_data[:, i*FLAGS.windows_size:(i+1)*FLAGS.windows_size])
            test_labels_feature.append(labels_feature_data[:, i][:, :, 0])
            test_adj_fc_pre.append(adj_fc_pre_data[:, i * FLAGS.windows_size:(i + 1) * FLAGS.windows_size])
            test_adj_label.append(ydata[:, i])
    return adj_sc, adj_fc, adj_label, features_sc, features_fc, adj_fc_pre, labels_feature, test_adj_sc, test_adj_fc, test_adj_label, test_features_sc, test_features_fc, test_adj_fc_pre, test_labels_feature

