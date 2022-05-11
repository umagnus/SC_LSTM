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
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

from optimizer import OptimizerED
from input_data import loadFSData
from models import Discriminator, lstmGAN
from preprocessing import preprocess_graph, construct_feed_dict, getDataTradition
from paramaters import FLAGS
from matplotlib.pyplot import savefig


np.set_printoptions(suppress=True)

# Load data
sc_features, sc_adj, fc_adj, fc_features = loadFSData("G:\Data\SC_LSTM test\dMRI", "G:\Data\SC_LSTM test\REST1")
index = []
for i in range(len(sc_adj)):
    index.append(i)

time_length = 1200
window_length = FLAGS.window_length
node_number = 90
remove_length = 1
window_number = time_length - window_length
num_features = 90
features_nonzero = 890

features_fc, labels_feature, test_features_fc, test_labels_feature = getDataTradition(index, fc_features, node_number)

# rmse = 0
# mae = 0
# for i in range(node_number):
#     pred = []
#     svr_rbf = SVR(kernel='rbf', C=1)
#     svr_rbf.fit(features_fc[:, i].tolist(), labels_feature[:, i, 0].tolist())
#     for p in range(FLAGS.pred_size):
#         test_tmp = test_features_fc[:, i, p:]
#         for tmp in range(p):
#             test_tmp = np.concatenate((test_tmp, pred[tmp].reshape(pred[tmp].shape[0], 1)), axis=1)
#         pred.append(svr_rbf.predict(test_tmp.tolist()))
#         # print(np.sqrt(sum(pred[p+1]-test_labels_feature[:, i, p].tolist())**2)/len(test_labels_feature[:, i, p].tolist()))
#     pred = np.transpose(np.array(pred), axes=(1, 0))
#     rmse = rmse + sum(sum((pred - test_labels_feature[:, i].tolist())**2) / len(test_labels_feature[:, i].tolist()))/FLAGS.pred_size
#     mae = mae + sum(sum(abs(pred-test_labels_feature[:, i].tolist()))/len(test_labels_feature[:, i].tolist()))/FLAGS.pred_size
# print(rmse/node_number)
# print(mae/node_number)


def get_order(data):
    pmax = int(len(data) / 10)    #一般阶数不超过 length /10
    qmax = int(len(data) / 10)
    bic_matrix = []
    for p in range(pmax +1):
        temp = []
        for q in range(qmax+1):
            try:
                temp.append(ARIMA(data, order=(p, 1, q)).fit().bic)    # 将bic值存入二维数组
            except:
                temp.append(None)
        bic_matrix.append(temp)
    bic_matrix = pd.DataFrame(bic_matrix)   #将其转换成Dataframe 数据结构
    p,q = bic_matrix.stack().astype('float64').idxmin()        # 找出bic值最小对应的索引
    return p,q


# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()


rmse = 0
mae = 0
for i in range(node_number):
    sub_data = test_features_fc[:, i, :]
    sub_label = test_labels_feature[:, i, :]
    sub_ind = np.expand_dims(np.arange(sub_data.shape[1]), axis=1)
    pred = []
    for sub in range(10):
        # p, q = get_order(sub_data[sub])
        # df = pd.DataFrame(sub_data[sub])
        # df.index = df.index.astype('str')
        # df.columns = df.columns.astype('str')
        # draw_trend(pd.DataFrame(sub_data[sub]), 10)
        model = ARIMA(sub_data[sub], order=(1, 0, 0)).fit()
        pred.append(model.predict(sub_data.shape[1]+1, sub_data.shape[1]+FLAGS.pred_size-1))
        # plt.figure()
        # plt.plot(sub_data[sub])
        # plt.plot(pred[sub], label='pred')
        # plt.plot(sub_label[sub], label='truth')
        # plt.legend()
        # plt.show()
    pred = np.array(pred)
    print(pred.shape)
    rmse = rmse + sum(sum((pred - test_labels_feature[:10, i].tolist())**2) / len(test_labels_feature[:10, i].tolist()))/FLAGS.pred_size
    mae = mae + sum(sum(abs(pred-test_labels_feature[:10, i].tolist()))/len(test_labels_feature[:10, i].tolist()))/FLAGS.pred_size
    print(rmse)
    print(mae)
    # plt.figure()
    # plt.show()
print(rmse/node_number)
print(mae/node_number)