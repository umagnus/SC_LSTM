"""
Database and target design. Data for each subject was collected in six fMRI sessions:
Day 1
- resting-state session (REST1) , time_length: 1200
- working-memory task (REST_WM),  time_length: 405
- motor task (REST_MT), time_length: 284
- gambling task (REST_GB), time_length: 253
Day 2
- resting-state session (REST2), time_length: 1200
- language task (REST_LG), time_length: 316
- emotion task (REST_EM), time_length: 176
- social cognition task (REST_SO), time_length: 274
- relational processing task (REST_RE), time_length: 232
For identification, we used a set of connectivity matrices from one session for the database
and connectivity matrices from a second session acquired on a different day as the
target set. All possible combinations of database and target sessions are indicated
by the arrows connecting session pairs.
"""

import numpy as np
from scipy import io
import scipy.sparse as sp
from Parameters import *
import os
import gc


def MatToList(edge_weight):
    """
    convert the type(.mat) of edge weight to list
    :param edge_weight:
    :return: edge_weight_list
    """
    node_number = edge_weight.shape[0]
    # feature_number = int(node_number * (node_number - 1) / 2)
    edge_weight_list = np.arange(feature_number).astype('float32')
    k = 0
    for i in range(0, node_number):
        for j in range(i+1, node_number):
            edge_weight_list[k] = edge_weight[i, j]
            k = k+1
    return edge_weight_list


def EdgeIndex(edge_weight):
    """

    :param edge_weight:
    :return: edge_index
    """
    node_number = edge_weight.shape[0]
    # feature_number = int(node_number*(node_number-1)/2)
    edge_index = np.arange(feature_number*3).reshape((feature_number, 3)).astype('int32')
    k = 0
    for i in range(0, node_number):
        for j in range(i+1, node_number):
            edge_index[k, :] = [k+1, i+1, j+1]
            k = k+1
    return edge_index


def featureZScore(Features):
    """
    use Z-Score to standardize features
    :param Features:
    :return: Features ; Mu: mean value; Sigma: standard deviation
    """
    eps = 3e-16
    feature_number = Features.shape[1]
    Mu = np.arange(feature_number).astype('float32')
    Sigma = np.arange(feature_number).astype('float32')
    for i in range(0, feature_number):
        i_feature = Features[:, i]
        i_mean = np.mean(i_feature)
        i_std = np.std(i_feature)
        if i_std < eps:
            i_std = eps
        Mu[i] = i_mean
        Sigma[i] = i_std
        Features[:, i] = (i_feature - i_mean)/i_std
    return Features, Mu, Sigma


def featureNormalize(Features, Mu, Sigma):
    """
    normalize the features with calculated mean value and standard deviation
    :param Features:
    :param Mu: mean value
    :param Sigma: standard deviation
    :return:
    """
    feature_number = Features.shape[1]
    for i in range(0, feature_number):
        i_feature = Features[:, i]
        Features[:, i] = (i_feature - Mu[i])/Sigma[i]
    return Features


def loadData(Data_type, time_length):
    """

    :return: Features: pair of nodes pearson correlation 256 * (256 - 1) / 2
              Labels: index of subject
    """
    R1_dir = os.path.join(main_dir, Data_type)
    file_list = os.listdir(R1_dir)
    file_list.sort()

    window_number = time_length - window_length + 1 - remove_length
    window_number_total = window_number * subject_number
    # feature_number = int(node_number * (node_number - 1) / 2)
    Features = np.arange(window_number_total*feature_number).reshape((
        window_number_total, feature_number)).astype('float32')
    Labels = np.zeros(window_number_total*subject_number).reshape((
        window_number_total, subject_number)).astype('int')

    i = 0
    for filespath in file_list:
        subj_dir = os.path.join(R1_dir, filespath, RS_dir)
        subj_data = io.loadmat(subj_dir)
        print("reading data " + subj_dir)
        subj_mat_rs = subj_data['RS_Regressed']
        for w in range(0, window_number):
            w_start = w + remove_length
            w_end = w_start + window_length
            w_series = np.transpose(subj_mat_rs[w_start:w_end, :])
            w_edgeWeight = np.corrcoef(w_series)
            w_sort = w_edgeWeight.reshape((-1))
            thindex = int(threshold * w_sort.shape[0])
            thremax = w_sort[w_sort.argsort()[-1*thindex]]
            print(thremax)
            w_edgeWeight[w_edgeWeight >= thremax] = 1
            w_edgeWeight[w_edgeWeight < thremax] = 0
            w_edgeWeight = w_edgeWeight-sp.eye(w_edgeWeight.shape[0])
            out_mat_name = "SC_network "+str(threshold*100)+".mat"
            io.savemat(os.path.join(R1_dir, filespath, out_mat_name), {"SC": w_edgeWeight})

    return Features, Labels


def HCPNetwork(result_dir, type_dict, result_name):
    """

    :param result_dir:
    :param type_dict: data dictionary: {"type_name": time_length}
    :param result_name: storage name of dataset
    :return:
    """
    x_data = np.array([])
    y_data = np.array([])
    for data_type, time_length in type_dict.items():
        print("==========read the data of: {}============".format(data_type))
        x, y = loadData(data_type, time_length)
        x_data = np.concatenate((x_data, x), axis=0) if x_data.size else x
        y_data = np.concatenate((y_data, y), axis=0) if y_data.size else y
    print(x_data.shape)
    x_data = featureZScore(x_data)[0]
    i_result_path = '{}/{}_WL{}.npz'.format(result_dir, result_name, window_length)
    np.savez(i_result_path, x=x_data, y=y_data)


if __name__ == "__main__":
    rest1 = {
        "REST1": 1200
    }
    rest2 = {
        "REST2": 1200
    }
    task1 = {
        "REST_WM": 405,
        "REST_MT": 284,
        "REST_GB": 253
    }
    task2 = {
        "REST_LG": 316,
        "REST_EM": 176,
        "REST_SO": 274,
        "REST_RE": 232
    }
    loadData("", 1200)
    # HCPNetwork(Network_Dir, task1, "TASK1")
    # HCPNetwork(Network_Dir, task2, "TASK2")
    # a = None
    # b = np.array([0, 1])
    # c = np.array([[4, 5, 6], [6, 7, 8]])
    # d = np.array([2, 3])
    # a = np.concatenate((a, c), axis=0) if a else c
    # b = np.concatenate((b, d), axis=0)
    # print(a)
    # print(b)
    a = np.array([])
    print(a.size)

