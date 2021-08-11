class base:
    def __init__(self,parameters={}):
        return

    def fit(self,data):
        return data
    
    def transform(self,data):
        return data
    
    def getParameter(self,key,default,parameters):
        return default if key not in parameters.keys() else parameters[key]

    def __str__(self):
        return "BaseProcessor"