import numpy as np
from sklearn.linear_model import LinearRegression
import copy
import torch
from torchvision import datasets, transforms

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


