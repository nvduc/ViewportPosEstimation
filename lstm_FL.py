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


def load_dataset(datapath, uid, test_vid, seq_length, delay_length):
    # Load all user view data
    df = pd.read_csv("view_list.csv")
    vids = df.loc[df['uid'] == uid]['vid']
    # print(vids)
    df_all = {}
    for vid in vids:
        df_all[vid] = pd.read_csv("{}xyz_vid_{}_uid_{}.txt".format(datapath, vid, uid), sep='\t', names=['phi', 'theta'])
    max_phi, min_phi = 180, -180
    max_theta, min_theta = 90, -90
    x_all, y_all = [], []
    # train data
    for vid in df_all:
        if vid != test_vid:
            training_data = df_all[vid].iloc[:,0:1].values
            training_data = transform(training_data, min_phi, max_phi)
            x, y = sliding_windows(training_data, seq_length, delay_length)
            x_all.append(x)
            y_all.append(y)
    x_train = np.concatenate(x_all,0)
    y_train = np.concatenate(y_all,0)
    # test data
    test_set = df_all[test_vid].iloc[:,0:1].values
    test_data = transform(test_set, min_phi, max_phi)
    x_test, y_test = sliding_windows(test_data, seq_length, delay_length)
    return x_train, y_train, x_test, y_test
    

if __name__ == '__main__':

    # settings
    max_phi, min_phi = 180, -180
    max_theta, min_theta = 90, -90
    datapath = "./data/"
    uid_list = [0, 1]
    test_vid = 0
    seq_length, delay_length = 30, 30


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
    num_rounds = 10
    num_epochs = 2000
    learning_rate = 0.01
    input_size = 1
    hidden_size = 3
    num_layers = 1
    num_classes = 1

    # BUID MODEL
    global_model = LSTM(num_classes, input_size, hidden_size, num_layers)
    global_model.to(device)
    global_model.train()
    print(global_model)
    # copy weights
    global_weights = global_model.state_dict()

    # criterion
    criterion = torch.nn.MSELoss().to(cuda_dev)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    acc_result = []

    for rnd in range(num_rounds):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {rnd+1} |\n')

        global_model.train()

        for uid in uid_list:
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                outputs = local_model(trainX[uid])
                optimizer.zero_grad()
                # obtain the loss function
                loss = criterion(outputs, trainY[uid])
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print("G-round: %d, user #%d, Epoch: %d, loss: %1.5f, rmse: %.5f" % (rnd, uid, epoch, loss.item(), np.sqrt(loss.item())))
            
            w = local_model.state_dict()
            local_weights.append(copy.deepcopy(w))
            # update global weights (averaging)
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

