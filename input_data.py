import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
from scipy import io
from paramaters import FLAGS
import seaborn as sns
import matplotlib.pyplot as plt
import preprocessing
from sklearn import preprocessing


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def MatToList(edge_weight):
    """
    convert the type(.mat) of edge weight to list
    :param edge_weight:
    :return: edge_weight_list
    """
    node_number = edge_weight.shape[0]
    feature_number = int(node_number * (node_number - 1) / 2)
    edge_weight_list = np.arange(feature_number).astype('float32')
    k = 0
    for i in range(0, node_number):
        for j in range(i+1, node_number):
            edge_weight_list[k] = edge_weight[i, j]
            k = k+1
    return edge_weight_list

def loadFSData(SC_path, FC_path):
    SC_list = os.listdir(SC_path)
    FC_list = os.listdir(FC_path)
    SC_list.sort()
    FC_list.sort()
    # SC_dir = "T1w\Diffusion\\network_DTI.mat"
    SC_dir = "DTI_connectivity_voxel_norm.mat"
    FC_dir = "aal_RS_Regressed.mat"
    feature = []
    adj = []
    label = []
    fc_adj = []
    fc_features = []
    for SCfiles in SC_list:
        subj_dir = os.path.join(SC_path, SCfiles, SC_dir)
        subj_data = io.loadmat(subj_dir)
        print("reading data " + subj_dir)
        # subj_mat_sc_all = subj_data['network_DTI']
        # subj_mat_sc = subj_mat_sc_all[0, 0]['CD'][0, 0]['matrix']
        subj_mat_sc = subj_data['connectivity']
        feature.append(np.identity(90).tolist())
        adj.append(subj_mat_sc)
    time_length = 1200
    window_length = FLAGS.window_length
    node_number = 90
    threshold = 0.8
    window_number = time_length - window_length
    for FCfiles in FC_list:
        subj_dir = os.path.join(FC_path, FCfiles, FC_dir)
        subj_data = io.loadmat(subj_dir)
        print("reading data " + subj_dir)
        subj_mat_fc = subj_data['RS_Regressed']
        subj_mat_fc_list = subj_mat_fc.reshape((-1))
        subj_mat_fc_new = (subj_mat_fc-min(subj_mat_fc_list))/(max(subj_mat_fc_list)-min(subj_mat_fc_list))

        # ind=[]
        # for indx in range(len(subj_mat_fc_new[0])):
            # ind.append(indx)
        # plt.plot(ind, subj_mat_fc_new[0], color='red', lw=2.5)
        # plt.plot(ind, subj_mat_fc_new[1], color='blue', lw=2.5)
        # plt.plot(ind, subj_mat_fc_new[2], color='green', lw=2.5)
        # plt.plot(ind, subj_mat_fc_new[3], color='brown', lw=2.5)
        # plt.plot(ind, subj_mat_fc_new[4], color='darkblue', lw=2.5)
        # plt.show()
        # subj_mat_fc_new = preprocessing.scale(subj_mat_fc)
        i_Features = np.arange(window_number*node_number*window_length).reshape((window_number, node_number, window_length)).astype('float32')
        i_adj = np.arange(window_number*node_number*node_number).reshape((window_number, node_number, node_number)).astype('float32')
        for w in range(0, window_number, FLAGS.remove_length):
            w_start = w
            w_end = w_start + window_length
            w_series = np.transpose(subj_mat_fc_new[w_start:w_end, :])
            w_edgeWeight = np.corrcoef(w_series)
            #edgeWeight_list = MatToList(w_edgeWeight)
            edgeWeight_list = w_edgeWeight.reshape((-1))
            thindex = int(threshold * edgeWeight_list.shape[0])
            thremax = edgeWeight_list[edgeWeight_list.argsort()[-1*thindex]]
            w_edgeWeight[w_edgeWeight < 0] = 0
            # w_edgeWeight = (w_edgeWeight-min(edgeWeight_list))/(1-min(edgeWeight_list))
            # w_edgeWeight[w_edgeWeight >= thremax] = 1
            # w_edgeWeight[w_edgeWeight < thremax] = 0
            i_adj[w, :, :] = w_edgeWeight
            i_Features[w, :, :] = np.transpose(subj_mat_fc_new[w_start:w_end, :])

            # if w % 40 == 0:
            #     fig0 = plt.figure(figsize=(6, 4))
            #     ax = fig0.add_axes([0.2, 0.07, 0.6, 0.9], facecolor='white')
            #     cbar_ax = fig0.add_axes([0.85, 0.07, 0.05, 0.88])
            #     fig_pred = sns.heatmap(np.reshape(w_edgeWeight, (90, 90)), ax=ax, cbar_ax=cbar_ax,
            #                        cmap='YlGnBu')
            #     heatmap_pred = fig_pred.get_figure()
            #     heatmap_pred.savefig('./heatmap_new_2/' + FCfiles+  'w=' + str(w) + '.jpg', dpi=400)
        fc_adj.append(i_adj)
        fc_features.append(i_Features)
    return feature, adj, fc_adj, fc_features


