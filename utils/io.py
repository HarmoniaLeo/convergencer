import numpy as np
import pandas as pd

def readNpy(dir):
    return pd.DataFrame(np.load(dir))
        
def readCsv(dir,id):
    if id is None:
        return pd.read_csv(dir)
    else:
        return pd.read_csv(dir,index_col=[id])

def readTxt(dir,delimiter):
    return pd.DataFrame(np.loadtxt(dir,delimiter=delimiter))

def readData(data,delimiter,id):
    if type(data)==str:
        if ".txt" in data:
            data = readTxt(data,delimiter)
        elif ".csv" in data:
            data = readCsv(data,id)
        elif ".npy" in dir:
            data = readNpy(data)
        else:
            raise Exception("Unsupported data type. ")
    elif type(data) == np.array:
        data = pd.DataFrame(data)
    elif type(data) != pd.DataFrame:
        raise Exception("Unsupported data type. ")
    if not (id is None):
        data = data.drop(id,axis=1)
    print("Get data with {0} rows and {1} cols. ".format(data.shape[0],data.shape[1]))
    return data

def readLabel(data,label,delimiter,labelId):
    if type(label)==str:
        if ".txt" in label:
            label = readTxt(label,delimiter)
        elif ".csv" in label:
            label = readCsv(label,labelId)
        elif ".npy" in label:
            label = readNpy(label)
    elif type(label) == int:
        label = data.columns[label]
        data = data.drop(label,axis=1)
        label = data[label]
    elif type(label) == np.array:
        label = pd.DataFrame(label)
    elif type(label) != pd.DataFrame:
        raise Exception("Unsupported label type. ")
    if not (labelId is None):
        label = label.drop(labelId,axis=1)
    print("Get features with {0} rows and {1} cols and labels with {2} rows. ".format(data.shape[0],data.shape[1],label.shape[1]))
    return data,label