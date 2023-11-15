

import numpy as np
import pandas as pd
import torch
import sys

def getCommandLineArgs():
    expectedKeys = ["epoch","dataLocation","outputLocation","lRate"]
    try:
        allArgs = {}
        for x in expectedKeys:
            allArgs[x] = sys.argv[sys.argv.index(x)+1]
        return allArgs

    except:
        print("Error: command line argument not given correclty!!!")

def getXYSet(dataList,dataDict,masterClassOrder):
    X, Y = [],[]
    for rec in dataList:
        y = np.zeros(len(masterClassOrder))
        y[masterClassOrder.index(rec[1])]= 1
        #imgNm = rec[0].split(".")[0].replace("_","-")
        imgNm = rec[0]
        if(imgNm in dataDict):
            x = dataDict[imgNm]
            X.append(x)
            Y.append(y)
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))
