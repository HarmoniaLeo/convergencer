import numpy as np
'''
def sigmoid(parameter):
    return 1/(1+np.exp(-parameter))

def intClass(parameter,min,max=None):
    if max is None:
        function=np.exp
        ranging=1
    else:
        function=sigmoid
        ranging=max-min
    parameter=function(parameter)*ranging+min
    return round(parameter)

print(intClass(2.6,1))
'''
class base:
    
    def print(self):
        print(self.myStr)

class son(base):
    def __init__(self):
        self.myStr="test"

son().print()