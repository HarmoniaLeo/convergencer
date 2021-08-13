import numpy as np

def sigmoid(parameter):
    return 1/(1+np.exp(-parameter))

def chooseClass(classlist,parameter):
    basing=1//len(classlist)
    return classlist[(parameter+1)/2//basing]

def intClass(parameter,min,max=None):
    if max is None:
        function=np.exp
        ranging=1
    else:
        function=np.sigmoid
        ranging=max-min
    parameter=function(parameter)*ranging+min
    return round(parameter)