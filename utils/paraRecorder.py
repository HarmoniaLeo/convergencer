class paraRecorder:
    def __init__(self):
        self.paraDict={}
        self.paraPath=[]

    def __getitem__(self,key):
        para=self.paraDict
        for k in self.paraPath:
            para=para[k]
        return para[key]
    
    def __setitem__(self,key,value):
        para=self.paraDict
        for k in self.paraPath:
            para=para[k]
        para[key]=value

    def getPara(self,key):
        return self.paraDict[key]