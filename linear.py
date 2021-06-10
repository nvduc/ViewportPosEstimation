#!/usr/bin/env python
# coding: utf-8
# using lstm-based network at individual users
# Incorporate latitute values --> Getting final prediction performance


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from os import listdir
from os.path import isfile, join
import time

from models import *
from functions import *
from common import *

if __name__ == '__main__':

    args = args_parser()
    print(args)
     # settings
    seq_length, delay_length = args.seq_length, 30
    acc_result = []
     # load train data
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    trainX, trainY, testX, testY = {}, {}, {}, {}

    x_train_theta, y_train_theta, x_test_theta, y_test_theta = {}, {}, {}, {}
    trainX_theta, trainY_theta, testX_theta, testY_theta = {}, {}, {}, {}
    
    for uid in uid_list:
        x_train[uid], y_train[uid], x_test[uid], y_test[uid], x_train_theta[uid], y_train_theta[uid], x_test_theta[uid], y_test_theta[uid] = load_dataset_full(datapath, uid, test_vid, seq_length, delay_length)

    rmse_all = []
    for runid in range(1):
        rmse_run = []
        for uid in uid_list:
            
            test_gt = inverse_transform(np.array(y_test[uid]).flatten(), min_phi, max_phi)
            test_gt_theta = inverse_transform(np.array(y_test_theta[uid]).flatten(), min_theta, max_theta)
            test_pred = inverse_transform(LR(x_test[uid], delay_length), min_phi, max_phi)
            test_pred_theta = inverse_transform(LR(x_test_theta[uid], delay_length), min_theta, max_theta)


            test_err = []
            for i in range(len(test_pred)):
                test_err.append(ang_dist(test_gt[i], test_gt_theta[i], test_pred[i], test_pred_theta[i]))
            test_result = [test_gt, test_gt_theta, test_pred, test_pred_theta, test_err]
            df_test_result = pd.DataFrame(np.array(test_result).transpose(), columns=['phi','theta','est_phi','est_theta', 'pred_err'])
            df_test_result.to_csv('result/linear_result_sw_{}_horiz_{}_uid_{}.csv'.format(seq_length, delay_length, uid), index=None)

            rmse_test = np.sqrt(np.mean(np.array(test_err)*np.array(test_err)))
            rmse_run.append(rmse_test)
        rmse_all.append(rmse_run)

# record results
cols = []
for uid in uid_list:
    cols.append('user #{}'.format(uid))
df = pd.DataFrame(rmse_all, columns=cols)
df.to_csv("result/linear_rmse_sw_{}_horiz_{}.csv".format(seq_length, delay_length), index=None)

