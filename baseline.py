# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
import os
from scipy import io
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size-seq_len-pre_len:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    return trainX, trainY, testX, testY


###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b) / la.norm(a)
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1 - F_norm, r2, var

FC_path = "G:\Data\REST1"
FC_list = os.listdir(FC_path)
FC_list.sort()
FC_list = FC_list
FC_dir = "aal_RS_Regressed.mat"
rmse_all = 0
mae_all = 0
fc_rmse = 0
fc_mae = 0
index = 0

fig0 = plt.figure(figsize=(6, 4))
ax = fig0.add_axes([0.2, 0.07, 0.6, 0.9], facecolor='white')
cbar_ax = fig0.add_axes([0.85, 0.07, 0.05, 0.88])

for FCfiles in FC_list:
    subj_dir = os.path.join(FC_path, FCfiles, FC_dir)
    subj_data = io.loadmat(subj_dir)
    print("reading data " + subj_dir)
    subj_mat_fc = subj_data['RS_Regressed']
    subj_mat_fc_list = subj_mat_fc.reshape((-1))
    # data = (subj_mat_fc-np.mean(subj_mat_fc))/np.std(subj_mat_fc_list)
    data = (subj_mat_fc - min(subj_mat_fc_list)) / (max(subj_mat_fc_list) - min(subj_mat_fc_list))
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    train_rate = 0.8
    seq_len = 40
    pre_len = 40
    trainX, trainY, testX, testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
    method = 'ARIMA'  ####HA or SVR or ARIMA

    ########### HA #############
    if method == 'HA':
        result = []
        for i in range(len(testX)):
            a = testX[i]
            a1 = np.mean(a, axis=0)
            result.append(a1.repeat(pre_len, axis=0))
        result1 = np.array(result)
        result2 = np.transpose(result1, axes=(0, 2, 1))
        # print(result2.shape)
        testX2 = np.transpose(np.array(testX), axes=(0, 2, 1))[:, :, -(80-pre_len):]
        # print(testX2.shape)
        result2 = np.concatenate([testX2, result2], axis=2)
        # print(result2.shape)
        fc_label = []
        for fc in range(result2.shape[0]):
            fc_label.append(np.corrcoef(result2[fc]))
        fc_label = np.array(fc_label)
        fc_label[fc_label < 0] = 0
        fc_label = np.reshape(fc_label, [-1, num_nodes])
        result1 = np.reshape(result1, [-1, num_nodes])
        testY1 = np.array(testY)
        testY2 = np.transpose(testY1, axes=(0, 2, 1))
        testY2 = np.concatenate([testX2, testY2], axis=2)
        fc_pred = []
        for fc in range(testY2.shape[0]):
            fc_pred.append(np.corrcoef(testY2[fc]))
        fc_pred = np.array(fc_pred)
        fc_pred[fc_pred < 0] = 0
        fc_pred = np.reshape(fc_pred, [-1, num_nodes])
        testY1 = np.reshape(testY1, [-1, num_nodes])
        rmse1, mae1, accuracy, r2, var = evaluation(testY1, result1)
        rmse2, mae2, acc2, r3, var2 = evaluation(fc_label, fc_pred)
        rmse_all = rmse_all + rmse1
        mae_all = mae_all + mae1
        fc_rmse = fc_rmse + rmse2
        fc_mae = fc_mae + mae2
        print('HA_rmse:%r' % rmse1,
              'HA_mae:%r' % mae1,
              'HA_acc:%r' % rmse2,
              'HA_r2:%r' % mae2,
              'HA_var:%r' % var)

    ############ SVR #############
    if method == 'SVR':
        total_rmse, total_mae, total_acc, result = [], [], [], []
        for i in range(num_nodes):
            data1 = np.mat(data)
            a = data1[:, i]
            a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
            a_X = np.array(a_X)
            a_X = np.reshape(a_X, [-1, seq_len])
            a_Y = np.array(a_Y)
            a_Y = np.reshape(a_Y, [-1, pre_len])
            a_Y = np.mean(a_Y, axis=1)
            t_X = np.array(t_X)
            t_X = np.reshape(t_X, [-1, seq_len])
            t_Y = np.array(t_Y)
            t_Y = np.reshape(t_Y, [-1, pre_len])

            svr_model = SVR(kernel='linear')
            svr_model.fit(a_X, a_Y)
            pre = svr_model.predict(t_X)
            pre = np.array(np.transpose(np.mat(pre)))
            pre = pre.repeat(pre_len, axis=1)
            result.append(pre)
        result1 = np.array(result)
        result2 = np.transpose(result1, axes=(1, 0, 2))
        testX2 = np.transpose(np.array(testX), axes=(0, 2, 1))[:, :, -(80-pre_len):]
        result2 = np.concatenate([testX2, result2], axis=2)
        fc_label = []
        for fc in range(result2.shape[0]):
            fc_label.append(np.corrcoef(result2[fc]))
        fc_label = np.array(fc_label)
        fc_label[fc_label < 0] = 0
        fc_label = np.reshape(fc_label, [-1, num_nodes])
        result1 = np.reshape(result1, [num_nodes, -1])
        result1 = np.transpose(result1)
        testY1 = np.array(testY)
        testY2 = np.transpose(testY1, axes=(0, 2, 1))
        testY2 = np.concatenate([testX2, testY2], axis=2)
        fc_pred = []
        for fc in range(testY2.shape[0]):
            fc_pred.append(np.corrcoef(testY2[fc]))
        fc_pred = np.array(fc_pred)
        fc_pred[fc_pred < 0] = 0
        print(fc_pred.shape)
        print(fc_label.shape)
        if index == 3:
            for ind in range(fc_pred.shape[0]):
                fig_pred = sns.heatmap(np.reshape(fc_pred[ind], (90, 90)), ax=ax, cbar_ax=cbar_ax,
                                        cmap='YlGnBu', vmin=0, vmax=1)
                heatmap_pred = fig_pred.get_figure()
                heatmap_pred.savefig(
                    './heatmap_new_4/' + 'SVR_index=' + str(ind) + '.jpg', dpi=400)
        fc_pred = np.reshape(fc_pred, [-1, num_nodes])

        testY1 = np.reshape(testY1, [-1, num_nodes])
        total = np.mat(total_acc)
        total[total < 0] = 0
        rmse1, mae1, acc1, r2, var = evaluation(testY1, result1)
        rmse2, mae2, acc2, r3, var2 = evaluation(fc_label, fc_pred)
        rmse_all = rmse_all + rmse1
        mae_all = mae_all + mae1
        fc_rmse = fc_rmse + rmse2
        fc_mae = fc_mae + mae2
        # if index == 3:
        #     label_graph = fc_label[i][:, 39].reshape((FLAGS.batch_size, 8100))
        #     pred_graph = fc_pred[1][:, 39]
        #     draw_graph(label_graph, pred_graph, 0, epoch, i, 39, index, ax, cbar_ax)
        print('SVR_rmse:%r' % rmse1,
              'SVR_mae:%r' % mae1,
              'SVR_fc_rmse:%r' % rmse2,
              'SVR_fc_mae:%r' % mae2,
              'SVR_var:%r' % var)

    ######## ARIMA #########
    if method == 'ARIMA' and index==3:
        rmse, mae, acc, r2, var, pred, ori, result = [], [], [], [], [], [], [], []
        for i in range(num_nodes):
            data1 = np.mat(data)
            a = data1[:, i]
            a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
            a_X = np.array(a_X)
            a_X = np.reshape(a_X, [-1, seq_len])
            a_Y = np.array(a_Y)
            a_Y = np.reshape(a_Y, [-1, pre_len])
            a_Y = np.mean(a_Y, axis=1)
            t_X = np.array(t_X)
            t_X = np.reshape(t_X, [-1, seq_len])
            t_Y = np.array(t_Y)
            t_Y = np.reshape(t_Y, [-1, pre_len])

            pre = []
            for ind in range(t_X.shape[0]):
                model = ARIMA(t_X[ind], order=[1, 0, 0])
                properModel = model.fit()
                predict_ts = properModel.predict(seq_len+1, seq_len+pre_len-1)
                print(predict_ts.shape)
                er_rmse, er_mae, er_acc, r2_score, var_score = evaluation(predict_ts, t_Y[ind])
                pre.append(predict_ts)
                rmse.append(er_rmse)
                mae.append(er_mae)
                acc.append(er_acc)
                r2.append(r2_score)
                var.append(var_score)
            pre = np.array(pre)
            result.append(pre)
        result1 = np.array(result)
        result2 = np.transpose(result1, axes=(1, 0, 2))
        testX2 = np.transpose(np.array(testX), axes=(0, 2, 1))[:, :, -(80-pre_len):]
        result2 = np.concatenate([testX2, result2], axis=2)
        fc_label = []
        for fc in range(result2.shape[0]):
            fc_label.append(np.corrcoef(result2[fc]))
        fc_label = np.array(fc_label)
        fc_label[fc_label < 0] = 0
        fc_label = np.reshape(fc_label, [-1, num_nodes])
        result1 = np.reshape(result1, [num_nodes, -1])
        result1 = np.transpose(result1)
        testY1 = np.array(testY)
        testY2 = np.transpose(testY1, axes=(0, 2, 1))
        testY2 = np.concatenate([testX2, testY2], axis=2)
        fc_pred = []
        for fc in range(testY2.shape[0]):
            fc_pred.append(np.corrcoef(testY2[fc]))
        fc_pred = np.array(fc_pred)
        fc_pred[fc_pred < 0] = 0
        if index == 3:
            for ind in range(fc_pred.shape[0]):
                fig_pred = sns.heatmap(np.reshape(fc_pred[ind], (90, 90)), ax=ax, cbar_ax=cbar_ax,
                                        cmap='YlGnBu', vmin=0, vmax=1)
                heatmap_pred = fig_pred.get_figure()
                heatmap_pred.savefig(
                    './heatmap_new_4/' + 'ARIMA_index=' + str(ind) + '.jpg', dpi=400)

        fc_pred = np.reshape(fc_pred, [-1, num_nodes])
        testY1 = np.reshape(testY1, [-1, num_nodes])
        rmse2, mae2, acc2, r3, var2 = evaluation(fc_label, fc_pred)
        fc_rmse = fc_rmse + rmse2
        fc_mae = fc_mae + mae2
        print('arima_rmse:%r' % (np.mean(rmse)),
              'arima_mae:%r' % (np.mean(mae)),
              'arima_fc_rmse:%r' % rmse2,
              'arima_fc_mae:%r' % mae2)

    ########### ST #############
    if method == 'ST':
        result = []
        for i in range(len(testX)):
            a = testX[i]
            a1 = a[-1]
            result.append(a1.repeat(pre_len, axis=0))
        result1 = np.array(result)
        result2 = np.transpose(result1, axes=(0, 2, 1))
        # print(result2.shape)
        testX2 = np.transpose(np.array(testX), axes=(0, 2, 1))[:, :, -(80-pre_len):]
        # print(testX2.shape)
        result2 = np.concatenate([testX2, result2], axis=2)
        # print(result2.shape)
        fc_label = []
        for fc in range(result2.shape[0]):
            fc_label.append(np.corrcoef(result2[fc]))
        fc_label = np.array(fc_label)
        fc_label[fc_label < 0] = 0
        fc_label = np.reshape(fc_label, [-1, num_nodes])
        result1 = np.reshape(result1, [-1, num_nodes])
        testY1 = np.array(testY)
        testY2 = np.transpose(testY1, axes=(0, 2, 1))
        testY2 = np.concatenate([testX2, testY2], axis=2)
        fc_pred = []
        for fc in range(testY2.shape[0]):
            fc_pred.append(np.corrcoef(testY2[fc]))
        fc_pred = np.array(fc_pred)
        fc_pred[fc_pred < 0] = 0
        fc_pred = np.reshape(fc_pred, [-1, num_nodes])
        testY1 = np.reshape(testY1, [-1, num_nodes])
        rmse1, mae1, accuracy, r2, var = evaluation(testY1, result1)
        rmse2, mae2, acc2, r3, var2 = evaluation(fc_label, fc_pred)
        rmse_all = rmse_all + rmse1
        mae_all = mae_all + mae1
        fc_rmse = fc_rmse + rmse2
        fc_mae = fc_mae + mae2
        print('HA_rmse:%r' % rmse1,
              'HA_mae:%r' % mae1,
              'HA_acc:%r' % rmse2,
              'HA_r2:%r' % mae2,
              'HA_var:%r' % var)

    index += 1


print(rmse_all)
print(mae_all)
print(fc_rmse)
print(fc_mae)

# path = r'data/los_speed.csv'
# data = pd.read_csv(path)

# time_len = data.shape[0]
# num_nodes = data.shape[1]
# train_rate = 0.8
# seq_len = 12
# pre_len = 3
# trainX, trainY, testX, testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
# method = 'HA'  ####HA or SVR or ARIMA
