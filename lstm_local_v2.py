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
    #train settings
    num_rounds = args.num_rounds
    num_epochs = args.num_epochs
    learning_rate = 0.01
    input_size = 1
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_classes = 1

    # check gpu/cuda support
    cuda_dev = None
    if torch.cuda.is_available():
        cuda_dev = torch.device('cuda')
    criterion = torch.nn.MSELoss()

    acc_result = []
     # load train data
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    trainX, trainY, testX, testY = {}, {}, {}, {}

    x_train_theta, y_train_theta, x_test_theta, y_test_theta = {}, {}, {}, {}
    trainX_theta, trainY_theta, testX_theta, testY_theta = {}, {}, {}, {}
    
    for uid in uid_list:
        x_train[uid], y_train[uid], x_test[uid], y_test[uid], x_train_theta[uid], y_train_theta[uid], x_test_theta[uid], y_test_theta[uid] = load_dataset_full(datapath, uid, test_vid, seq_length, delay_length)
        
        if cuda_dev != None:
            trainX[uid] = Variable(torch.Tensor(np.array(x_train[uid])).cuda())
            trainY[uid] = Variable(torch.Tensor(np.array(y_train[uid])).cuda())
            testX[uid] = Variable(torch.Tensor(np.array(x_test[uid])).cuda())
            testY[uid] = Variable(torch.Tensor(np.array(y_test[uid])).cuda())

            trainX_theta[uid] = Variable(torch.Tensor(np.array(x_train_theta[uid])).cuda())
            trainY_theta[uid] = Variable(torch.Tensor(np.array(y_train_theta[uid])).cuda())
            testX_theta[uid] = Variable(torch.Tensor(np.array(x_test_theta[uid])).cuda())
            testY_theta[uid] = Variable(torch.Tensor(np.array(y_test_theta[uid])).cuda())
        else:
            trainX[uid] = Variable(torch.Tensor(np.array(x_train[uid])))
            trainY[uid] = Variable(torch.Tensor(np.array(y_train[uid])))
            testX[uid] = Variable(torch.Tensor(np.array(x_test[uid])))
            testY[uid] = Variable(torch.Tensor(np.array(y_test[uid])))

            trainX_theta[uid] = Variable(torch.Tensor(np.array(x_train_theta[uid])))
            trainY_theta[uid] = Variable(torch.Tensor(np.array(y_train_theta[uid])))
            testX_theta[uid] = Variable(torch.Tensor(np.array(x_test_theta[uid])))
            testY_theta[uid] = Variable(torch.Tensor(np.array(y_test_theta[uid])))

    rmse_all = []
    for runid in range(args.num_run):
        rmse_run = []
        for uid in uid_list:
            lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
            lstm_theta = LSTM(num_classes, input_size, hidden_size, num_layers)
            train_pred, train_gt, test_pred, test_gt = train_and_eval_model(lstm, criterion, learning_rate, cuda_dev, num_epochs, trainX[uid], trainY[uid], testX[uid], testY[uid], min_phi, max_phi)
            train_pred_theta, train_gt_theta, test_pred_theta, test_gt_theta = train_and_eval_model(lstm_theta, criterion, learning_rate, cuda_dev, num_epochs, trainX[uid], trainY[uid], testX[uid], testY[uid], min_phi, max_phi)
            train_err, test_err = [], []
            for i in range(len(train_pred)):
                # print(train_gt[i], train_gt_theta[i], train_pred[i], train_pred_theta[i])
                train_err.append(ang_dist(train_gt[i], train_gt_theta[i], train_pred[i], train_pred_theta[i]))
            for i in range(len(test_pred)):
                test_err.append(ang_dist(test_gt[i], test_gt_theta[i], test_pred[i], test_pred_theta[i]))
            train_result = [train_gt, train_gt_theta, train_pred, train_pred_theta, train_err]
            df_train_result = pd.DataFrame(np.array(train_result).transpose(), columns=['phi','theta','est_phi','est_theta', 'pred_err'])
            df_train_result.to_csv('lstm_local_train_result_uid_{}_run_{}_{}.csv'.format(uid, runid, int(time.time())), index=None)
            test_result = [test_gt, test_gt_theta, test_pred, test_pred_theta, test_err]
            df_test_result = pd.DataFrame(np.array(test_result).transpose(), columns=['phi','theta','est_phi','est_theta', 'pred_err'])
            df_test_result.to_csv('lstm_local_train_result_uid_{}_run_{}_{}.csv'.format(uid, runid, int(time.time())), index=None)

            rmse_train = np.sqrt(np.mean(np.array(train_err)*np.array(train_err)))
            rmse_test = np.sqrt(np.mean(np.array(test_err)*np.array(test_err)))
            rmse_run.append(rmse_test)
        rmse_all.append(rmse_run)

# record results
cols = []
for uid in uid_list:
    cols.append('user #{}'.format(uid))
df = pd.DataFrame(rmse_all, columns=cols)
df.to_csv("result/lstm_local_rmse_{}.csv".format(int(time.time())), index=None)

