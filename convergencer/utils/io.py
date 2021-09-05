import numpy as np
import pandas as pd
import pickle

def readNpy(dir):
    return pd.DataFrame(np.load(dir))
        
def readCsv(dir):
    return pd.read_csv(dir)

def readTxt(dir,delimiter):
    return pd.DataFrame(np.loadtxt(dir,delimiter=delimiter))

def readData(data,delimiter=",",id=None):
    '''
    params:
        data: array/dataframe/[direction to the data file (.npy/.txt/.csv)]
        delimiter: for .txt
        id: col of index
    return:
        A dataframe with full data/features
    '''
    if type(data)==str:
        if ".txt" in data:
            data = readTxt(data,delimiter)
        elif ".csv" in data:
            data = readCsv(data)
        elif ".npy" in dir:
            data = readNpy(data)
        else:
            raise Exception("Unsupported data type. ")
    elif type(data) == np.ndarray:
        data = pd.DataFrame(data)
    elif type(data) != pd.DataFrame:
        raise Exception("Unsupported data type. ")
    if not (id is None):
        data=data.set_index(id)
    print("Get data with {0} rows and {1} cols. ".format(data.shape[0],data.shape[1]))
    return data

def readLabel(data,label=-1,delimiter=",",labelId=None):
    '''
    params:
        data: array/dataframe of full data
        label: [col of label in the full data] / [[array of the label]/[dataframe of the label]/[direction to the label data file (.npy/.txt/.csv)]]
        delimiter: for .txt
        id: col of index in the label data
    return:
        A dataframe with features
        A series with label
    '''
    if type(label)==str:
        if ".txt" in label:
            label = readTxt(label,delimiter)
        elif ".csv" in label:
            label = readCsv(label)
        elif ".npy" in label:
            label = readNpy(label)
        else:
            labelCol = label
            label = data[labelCol]
            data = data.drop(labelCol,axis=1)
    elif type(label) == int:
        labelCol = data.columns[label]
        label = data[labelCol]
        data = data.drop(labelCol,axis=1)
    elif type(label) == np.ndarray:
        label = pd.DataFrame(label)
    if type(label) == pd.DataFrame:
        if not (labelId is None):
            label=label.set_index(labelId)
        label=label.iloc[:,0]
    if type(label) != pd.Series:
        raise Exception("Unsupported label type. ")
    print("Get features with {0} rows and {1} cols and labels with {2} rows. ".format(data.shape[0],data.shape[1],label.shape[0]))
    return data,label

def readDict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def saveDict(parameters,path):
    with open(path, 'wb') as f:
        pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)