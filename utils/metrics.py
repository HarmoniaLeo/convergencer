import numpy as np

def mape(y_true,y_pred):
    loss = np.mean(np.abs((y_true - y_pred) / y_true),axis=-1)
    return loss

def mspe(y_true,y_pred):
    loss = np.mean(np.square(((y_true - y_pred) / y_true)),axis=-1)
    return loss