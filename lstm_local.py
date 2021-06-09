#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from os import listdir
from os.path import isfile, join

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
    if torch.cuda.is_available:
        cuda_dev = torch.device('cuda')
    acc_result = []
     # load train data
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    trainX, trainY, testX, testY = {}, {}, {}, {}
    for uid in uid_list:
        x_train[uid], y_train[uid], x_test[uid], y_test[uid] = load_dataset(datapath, uid, test_vid, seq_length, delay_length)
        trainX[uid] = Variable(torch.Tensor(np.array(x_train[uid])).cuda())
        trainY[uid] = Variable(torch.Tensor(np.array(y_train[uid])).cuda())
        testX[uid] = Variable(torch.Tensor(np.array(x_test[uid])).cuda())
        testY[uid] = Variable(torch.Tensor(np.array(y_test[uid])).cuda())

    rmse_all = []
    for runid in range(args.num_run):
        rmse_run = []
        for uid in uid_list:
            lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
            lstm.to(cuda_dev)
            criterion = torch.nn.MSELoss().to(cuda_dev)    # mean-squared error for regression
            optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
           
            # Train the model
            for epoch in range(num_epochs):
                outputs = lstm(trainX[uid])
                optimizer.zero_grad()
                
                # obtain the loss function
                loss = criterion(outputs, trainY[uid])
                
                loss.backward()
                optimizer.step()
                # if epoch % 100 == 0:
                #     print("Epoch: %d, loss: %1.5f, rmse: %.5f" % (epoch, loss.item(), np.sqrt(loss.item())))
            lstm.eval()
            test_pred = lstm(testX[uid])
            train_pred = lstm(trainX[uid])
            test_pred = inverse_transform(test_pred.data.cpu().numpy(), min_phi, max_phi)
            test_gt = inverse_transform(testY[uid].data.cpu().numpy(), min_phi, max_phi)
            rmse_tmp =  rmse(test_pred.flatten(), test_gt.flatten())
            rmse_run.append(rmse_tmp)
        rmse_all.append(rmse_run)
       # break
df = pd.DataFrame(rmse_all, columns=['user #0', 'user #1', 'user #2', 'user #3', 'user #4', 'user #5'])
df.to_csv("lstm_local.csv", index=None)
    # plt.plot(test_pred,'b')
    # plt.plot(test_gt, 'r')
    # plt.suptitle('Test Prediction')
    # plt.show()

