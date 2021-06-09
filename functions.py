import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import copy
import torch
from torchvision import datasets, transforms
import argparse

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help="number of rounds of training")
    parser.add_argument('--num_rounds', type=int, default=5,
                        help="number of global tranining rounds")
    parser.add_argument('--hidden_size', type=int, default=5,
                        help="LSTM's hidden size")
    parser.add_argument('--num_layers', type=int, default=1,
                        help="LSTM's number of layers")
    parser.add_argument('--seq_length', type=int, default=30,
                        help="History window size")
    parser.add_argument('--num_run', type=int, default=5,
                        help="Number of runs")

    args = parser.parse_args()
    return args
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def rmse(pred, gt):
    return np.sqrt(np.mean((pred-gt)*(pred-gt)))

def transform(X, min_X, max_X):
    return (X - min_X)/(max_X - min_X)
def inverse_transform(X, min_X, max_X):
    return X * (max_X - min_X) + min_X
def sliding_windows(data, seq_length, delay_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-delay_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length + delay_length - 1]

        ## calibrate data at boudaries ##
        flg = True
        for i in range(len(_x) - 1):
            if(np.absolute(_x[i] - _x[i+1]) > 0.5):
                flg = False
                break
        #         if _x[i] > _x[i+1]:
        #             for j in range(i+1, len(_x)):
        #                 _x[j] += 1
        #             _y = _y + 1
        #         else:
        #             for j in range(i+1, len(_x)):
        #                 _x[j] -= 1
        #             _y = _y - 1;
        #         break
        if flg:
            x.append(_x)
            y.append(_y)

    return np.array(x),np.array(y)

#### Last method ####
def LAST(x_test):
    y_pred = []
    for x in x_test:
        y_pred.append(x[-1])
    return np.array(y_pred)


#### Linear regression ####
def LR(x, pred_hori):
    y_pred = []
    x_train = np.arange(len(x[0])).reshape(-1, 1)
    x_test = np.array([len(x[0]) + pred_hori]).reshape(1, -1)

    # print(x_train, x_test)
    for y_train in x:
        # print(y_train)
        # train
        reg = LinearRegression()
        reg.fit(x_train, y_train)

        # predict
        y_pred.append(reg.predict(x_test))
    return np.array(y_pred)


