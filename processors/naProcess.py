from base import base
from utils.processing import impute_knn

class fillNa(base):
    def __init__(self, parameters):
        self.fillNaStrg=self.getParameter("strategy",None,parameters)
        self.fillNaValue=self.getParameter("values",None,parameters)
        self.fillNaK=self.getParameter("k",5,parameters)
        self.fillNaCols=self.getParameter("knn cols",{},parameters)
    
    def transform(self,data):
        print("Try to fill nan values. ")
        for key in data.columns:
            nanCount = data[key].isna().sum()
            if nanCount==0:
                continue
            if (self.fillNaStrg is None) or (key not in self.fillNaStrg.keys()):
                continue
            else:
                if ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "mode")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "mode")):
                    num = data[key].mode()
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "mean")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "mean")):
                    num = data[key].mean()
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "auto")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "auto")):
                    if data[key].dtype == object:
                        num = data[key].mode()
                        print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mode "+str(num))
                    else:
                        num = data[key].mean()
                        print("Fill {0} nans in col ".format(nanCount)+str(key)+" with mean "+str(num))
                elif ((type(self.fillNaStrg)==str) and (self.fillNaStrg == "value")) or ((type(self.fillNaStrg)==dict) and (self.fillNaStrg[key] == "value")):
                    num = self.fillNaValue[key]
                    print("Fill {0} nans in col ".format(nanCount)+str(key)+" with "+str(num))
                else:
                    raise Exception("Unsupported filling strategy. ")
                data[key] = data[key].fillna(num)
        return impute_knn(data,self.fillNaK,self.fillNaKCols)
    
    def __str__(self):
        return "fillNa"