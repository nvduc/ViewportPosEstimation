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

if __name__ == '__main__':

    datapath = "./data/"
    uid = 0
    test_vid = 0
    acc_result = []
    for uid in range(60):

        # Load all user view data
        df = pd.read_csv("view_list.csv")
        vids = df.loc[df['uid'] == uid]['vid']
        # print(vids)
        df_all = {}
        for vid in vids:
            df_all[vid] = pd.read_csv("{}xyz_vid_{}_uid_{}.txt".format(datapath, vid, uid), sep='\t', names=['phi', 'theta'])

        seq_length = 30
        delay_length = 30
        max_phi, min_phi = 180, -180
        max_theta, min_theta = 90, -90

        if(len(vids) > 1):
            for test_vid in vids:
                x_all, y_all = [], []
                for vid in df_all:
                    if vid != test_vid:
                        training_data = df_all[vid].iloc[:,0:1].values
                        training_data = transform(training_data, min_phi, max_phi)
                        x, y = sliding_windows(training_data, seq_length, delay_length)
                        x_all.append(x)
                        y_all.append(y)
                x_train = np.concatenate(x_all,0)
                y_train = np.concatenate(y_all,0)

                test_set = df_all[test_vid].iloc[:,0:1].values
                test_data = transform(test_set, min_phi, max_phi)
                x_test, y_test = sliding_windows(test_data, seq_length, delay_length)


                # In[12]:

                trainX = Variable(torch.Tensor(np.array(x_train)))
                trainY = Variable(torch.Tensor(np.array(y_train)))

                testX = Variable(torch.Tensor(np.array(x_test)))
                testY = Variable(torch.Tensor(np.array(y_test)))



                num_epochs = 2000
                learning_rate = 0.01

                input_size = 1
                hidden_size = 3
                num_layers = 1

                num_classes = 1

                lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
                criterion = torch.nn.MSELoss()    # mean-squared error for regression
                optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
                #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

                # setup GPU
                cuda_dev = None
                # print("---Has GPU: ", torch.cuda.is_available())
                if torch.cuda.is_available:
                    cuda_dev = torch.device('cuda')
                if cuda_dev != None:
                    lstm.to(cuda_dev)
                    criterion = torch.nn.MSELoss().to(cuda_dev)
                    trainX = Variable(torch.Tensor(np.array(x_train)).to(cuda_dev))
                    trainY = Variable(torch.Tensor(np.array(y_train)).to(cuda_dev))

                    testX = Variable(torch.Tensor(np.array(x_test)).cuda())
                    testY = Variable(torch.Tensor(np.array(y_test)).cuda())  
                # exit(1)

                # Train the model
                for epoch in range(num_epochs):
                    outputs = lstm(trainX)
                    optimizer.zero_grad()
                    
                    # obtain the loss function
                    loss = criterion(outputs, trainY)
                    
                    loss.backward()
                    optimizer.step()
                    # if epoch % 100 == 0:
                    #     print("Epoch: %d, loss: %1.5f, rmse: %.5f" % (epoch, loss.item(), np.sqrt(loss.item())))

                lstm.eval()
                test_pred = lstm(testX)
                train_pred = lstm(trainX)

                test_pred = inverse_transform(test_pred.data.cpu().numpy(), min_phi, max_phi)
                test_gt = inverse_transform(testY.data.cpu().numpy(), min_phi, max_phi)

                test_pred_last = inverse_transform(LAST(x_test), min_phi, max_phi)
                test_pred_LR = inverse_transform(LR(x_test, delay_length), min_phi, max_phi)
                print(uid, test_vid, rmse(test_pred.flatten(), test_gt.flatten()), rmse(test_pred_last.flatten(), test_gt.flatten()), rmse(test_pred_LR.flatten(), test_gt.flatten()))
                acc_result.append([uid, test_vid, rmse(test_pred.flatten(), test_gt.flatten()), rmse(test_pred_last.flatten(), test_gt.flatten()), rmse(test_pred_LR.flatten(), test_gt.flatten())])
        # break
df = pd.DataFrame(acc_result, columns=["uid", "vid", "LSTM", "LAST", "LR"])
df.to_csv("result_local_2.csv", index=None)
    # plt.plot(test_pred,'b')
    # plt.plot(test_gt, 'r')
    # plt.suptitle('Test Prediction')
    # plt.show()

