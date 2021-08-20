import os

traindirs=os.listdir("train")
testdirs=os.listdir("test")
for d in testdirs:
    if d in traindirs:
        print(d)