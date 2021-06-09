#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import copy

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from os import listdir
from os.path import isfile, join

from models import *
from functions import *
from common import *

import time

if __name__ == '__main__':

    args = args_parser()
    print(args)

    # settings
    seq_length, delay_length = args.seq_length, 30


    cuda_dev = None
    print("---Has GPU: ", torch.cuda.is_available())
    if torch.cuda.is_available:
        cuda_dev = torch.device('cuda')

    # load train data
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    trainX, trainY, testX, testY = {}, {}, {}, {}
    
    for uid in uid_list:
        x_train[uid], y_train[uid], x_test[uid], y_test[uid] = load_dataset(datapath, uid, test_vid, seq_length, delay_length)
        trainX[uid] = Variable(torch.Tensor(np.array(x_train[uid])).cuda())
        trainY[uid] = Variable(torch.Tensor(np.array(y_train[uid])).cuda())
        testX[uid] = Variable(torch.Tensor(np.array(x_test[uid])).cuda())
        testY[uid] = Variable(torch.Tensor(np.array(y_test[uid])).cuda())

    # train settings
    num_rounds = args.num_rounds
    num_epochs = args.num_epochs
    learning_rate = 0.01
    input_size = 1
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_classes = 1

    for runid in range(args.num_run):
        # BUID MODELs
        global_model = LSTM(num_classes, input_size, hidden_size, num_layers)
        global_model.to(cuda_dev)
        global_model.train()
        print(global_model)
        print("Number of params: ", count_parameters(global_model))
        # copy weights
        global_weights = global_model.state_dict()

        # local models
        local_model = {}
        for uid in uid_list:
            local_model[uid] = LSTM(num_classes, input_size, hidden_size, num_layers)
            local_model[uid].to(cuda_dev)
            local_model[uid].train()

        # criterion
        criterion = torch.nn.MSELoss().to(cuda_dev)

        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0
        acc_result = []
        rmse_round = {}
        for uid in uid_list:
            rmse_round[uid] = []
        local_rmse = np.zeros(len(uid_list))

        convergence = False
        rnd = 0
        while not convergence:
            local_weights, local_losses, acc = [], [], []
            print(f'\n | Global Training Round : {rnd+1} |\n')

            global_model.train()

            cnt = 0
            for uid in uid_list:
                gmodel_copy = copy.deepcopy(global_model)
                optimizer = torch.optim.Adam(gmodel_copy.parameters(), lr=learning_rate)
                for epoch in range(num_epochs):
                    outputs = gmodel_copy(trainX[uid])
                    optimizer.zero_grad()
                    # obtain the loss function
                    loss = criterion(outputs, trainY[uid])
                    loss.backward()
                    optimizer.step()
                    # if epoch % 100 == 0:
                    #     print("G-round: %d, user #%d, Epoch: %d, loss: %1.5f, rmse: %.5f" % (rnd+1, uid, epoch, loss.item(), np.sqrt(loss.item())))
                
                gmodel_copy.eval()
                test_pred = gmodel_copy(testX[uid])
                train_pred = gmodel_copy(trainX[uid])

                test_pred = inverse_transform(test_pred.data.cpu().numpy(), min_phi, max_phi)
                test_gt = inverse_transform(testY[uid].data.cpu().numpy(), min_phi, max_phi)

                print("G-round: %d, user #%d, rmse: %.2f" %(rnd + 1, uid, rmse(test_pred.flatten(), test_gt.flatten())))
                rmse_tmp = rmse(test_pred.flatten(), test_gt.flatten())
                rmse_round[uid].append(rmse_tmp)

                if rnd == 0 or rmse_round[uid][rnd] < local_rmse[uid]: ## Update local model ##
                    w = gmodel_copy.state_dict()
                    local_model[uid] = copy.deepcopy(gmodel_copy)
                    local_rmse[uid] = rmse_tmp
                    acc.append(rmse_tmp)
                else:
                    w = local_model[uid].state_dict()
                    acc.append(-local_rmse[uid])
                    cnt += 1

                local_weights.append(copy.deepcopy(w))
            # update global weights (averaging)
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)
            acc_result.append(acc)

            if cnt == len(uid_list):
                convergence = True

            rnd += 1

        t = int(time.time())
        df = pd.DataFrame(acc_result, columns = ['user #0', 'user #1', 'user #2', 'user #3', 'user #4', 'user #5'])
        df.to_csv("FL_v2_user_{}_HS_{}_round_{}_hidden_{}_{}.csv".format(len(uid_list), seq_length, num_rounds, hidden_size, t), index=None)

