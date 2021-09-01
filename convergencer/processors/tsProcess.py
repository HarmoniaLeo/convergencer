import pandas as pd
from processors.base import base

class tsToNum(base):
    def __init__(self, data, y=None, parameters={},verbose=1):
        '''
        parameters:
            {
                "formats": formats of each timestamp col
                    {
                        "col1":"format1",
                        "col2":"format2",
                        ...
                    }
            }
        '''
        self.verbose=verbose
        self.formats=self.getParameter("formats",{},parameters)

    def transform(self, data, y=None):
        if self.verbose==1:
            print("\n-------------------------Try to extrate numerical features from time-stamps-------------------------")
        data=data.copy()
        for key in self.formats.keys():
            s=pd.to_datetime(data[key],format=self.formats[key])
            data[str(key)+"_year"]=s.dt.year
            if self.verbose==1:
                print("Extract year from "+str(key))
            data[str(key)+"_month"]=s.dt.month
            if self.verbose==1:
                print("Extract month from "+str(key))
            data[str(key)+"_week"]=s.dt.week
            if self.verbose==1:
                print("Extract week from "+str(key))
            data[str(key)+"_day"]=s.dt.day
            if self.verbose==1:
                print("Extract day from "+str(key))
            data[str(key)+"_hour"]=s.dt.hour
            if self.verbose==1:
                print("Extract hour from "+str(key))
            data[str(key)+"_minute"]=s.dt.minute
            if self.verbose==1:
                print("Extract minute from "+str(key))
            data[str(key)+"_second"]=s.dt.second
            if self.verbose==1:
                print("Extract second from "+str(key))
            s-=s.min()
            data[str(key)+"_years"]=s.years
            if self.verbose==1:
                print("Extract years to go from "+str(key))
            data[str(key)+"_months"]=s.months
            if self.verbose==1:
                print("Extract months to go from "+str(key))
            data[str(key)+"_weeks"]=s.weeks
            if self.verbose==1:
                print("Extract weeks to go from "+str(key))
            data[str(key)+"_days"]=s.dt.days
            if self.verbose==1:
                print("Extract days to go from "+str(key))
            data[str(key)+"_hours"]=s.dt.hours
            if self.verbose==1:
                print("Extract hours to go from "+str(key))
            data[str(key)+"_minutes"]=s.dt.minutes
            if self.verbose==1:
                print("Extract minutes to go from "+str(key))
            data[str(key)+"_seconds"]=s.dt.seconds
            if self.verbose==1:
                print("Extract seconds to go from "+str(key))
        data=data.drop(self.formats.keys(),axis=1)
        return super().transform(data, y=y)

    def __str__(self):
        return "tsProcessor"