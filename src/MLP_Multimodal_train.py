

import numpy as np
import os
import glob 
import pandas as pd
import random
from sklearn.utils import shuffle
import pickle
import torch
import sys

def getAccuracyMx(Y,Yht,numClass=5):
    correctCounts = np.zeros(numClass)
    totalCounts = np.zeros(numClass)
    lenAll = len(Y)
    for i in range(lenAll):
        if(np.argmax(Y[i])== np.argmax(Yht[i])):
            correctCounts[np.argmax(Y[i])] += 1 
        totalCounts[np.argmax(Y[i])] += 1
    accPerClass = correctCounts/totalCounts
    return accPerClass


def getCommandLineArgs():
    expectedKeys = ["--epoch","--dataLocation","--outputLocation","--lRate","--valData","--testData","--modelName","--featureType","--bestmodelcrit","--datasetname","--trainData"]
    try:
        allArgs = {}
        for x in expectedKeys:
            if(x in sys.argv):
                allArgs[x] = sys.argv[sys.argv.index(x)+1]
        return allArgs

    except:
        print("Error: command line argument not given correclty!!!")

class NeuralNetworkSimpleV512_funnel(torch.nn.Module):
        def __init__(self, input_size,nClasses):
            super(NeuralNetworkSimpleV512_funnel, self).__init__()
            self.input_size = input_size
            self.h1Size  = 512
            self.h2Size  = 256
            self.h3Size  = 128
            self.h4Size  = 64
            self.h5Size  = 32
            self.h6Size  = 16
            self.outSize  = nClasses
            self.fc1 = torch.nn.Linear(self.input_size, self.h1Size)
            self.fc2 = torch.nn.Linear(self.h1Size, self.h2Size)
            self.fc3 = torch.nn.Linear(self.h2Size, self.h3Size)
            self.fc4 = torch.nn.Linear(self.h3Size, self.h4Size)
            self.fc5 = torch.nn.Linear(self.h4Size, self.h5Size)
            self.fc6 = torch.nn.Linear(self.h5Size, self.h6Size)
            self.fcOut = torch.nn.Linear(self.h6Size, self.outSize)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            h4 = self.relu(self.fc4(h3))
            h5 = self.relu(self.fc5(h4))
            h6 = self.relu(self.fc6(h5))
            output = self.fcOut(h6)
            output = self.sigmoid(output)
            return output
        

class NeuralNetworkSimpleV64(torch.nn.Module):
        def __init__(self, input_size,nClasses):
            super(NeuralNetworkSimpleV64, self).__init__()
            self.input_size = input_size
            self.h1Size  = 512
            self.h2Size  = 256
            self.h3Size  = 128
            self.h4Size  = 64
            self.h5Size  = 64
            self.h6Size  = 64
            self.outSize  = nClasses
            self.fc1 = torch.nn.Linear(self.input_size, self.h1Size)
            self.fc2 = torch.nn.Linear(self.h1Size, self.h2Size)
            self.fc3 = torch.nn.Linear(self.h2Size, self.h3Size)
            self.fc4 = torch.nn.Linear(self.h3Size, self.h4Size)
            self.fc5 = torch.nn.Linear(self.h4Size, self.h5Size)
            self.fc6 = torch.nn.Linear(self.h5Size, self.h6Size)
            self.fcOut = torch.nn.Linear(self.h6Size, self.outSize)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            h4 = self.relu(self.fc4(h3))
            h5 = self.relu(self.fc5(h4))
            h6 = self.relu(self.fc6(h5))
            output = self.fcOut(h6)
            output = self.sigmoid(output)
            return output
            
class NeuralNetworkSimpleV512(torch.nn.Module):
        def __init__(self, input_size,nClasses):
            super(NeuralNetworkSimpleV512, self).__init__()
            self.input_size = input_size
            self.h1Size  = 512
            self.h2Size  = 512
            self.h3Size  = 512
            self.h4Size  = 512
#            self.h5Size  = 512
#            self.h6Size  = 512
            self.outSize  = nClasses
            self.fc1 = torch.nn.Linear(self.input_size, self.h1Size)
            self.fc2 = torch.nn.Linear(self.h1Size, self.h2Size)
            self.fc3 = torch.nn.Linear(self.h2Size, self.h3Size)
            self.fc4 = torch.nn.Linear(self.h3Size, self.h4Size)
#            self.fc5 = torch.nn.Linear(self.h4Size, self.outSize)
#            self.fc6 = torch.nn.Linear(self.h5Size, self.h6Size)
            self.fcOut = torch.nn.Linear(self.h4Size, self.outSize)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            h4 = self.relu(self.fc4(h3))
#            h5 = self.relu(self.fc5(h4))
#            h6 = self.relu(self.fc6(h5))
            output = self.fcOut(h4)
            output = self.sigmoid(output)
            return output

class NeuralNetworkSimpleV256(torch.nn.Module):
        def __init__(self, input_size,nClasses):
            super(NeuralNetworkSimpleV256, self).__init__()
            self.input_size = input_size
            self.h1Size  = 512
            self.h2Size  = 512
            self.h3Size  = 256
            self.h4Size  = 256
#            self.h5Size  = 512
#            self.h6Size  = 512
            self.outSize  = nClasses
            self.fc1 = torch.nn.Linear(self.input_size, self.h1Size)
            self.fc2 = torch.nn.Linear(self.h1Size, self.h2Size)
            self.fc3 = torch.nn.Linear(self.h2Size, self.h3Size)
            self.fc4 = torch.nn.Linear(self.h3Size, self.h4Size)
#            self.fc5 = torch.nn.Linear(self.h4Size, self.outSize)
#            self.fc6 = torch.nn.Linear(self.h5Size, self.h6Size)
            self.fcOut = torch.nn.Linear(self.h4Size, self.outSize)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            h4 = self.relu(self.fc4(h3))
#            h5 = self.relu(self.fc5(h4))
#            h6 = self.relu(self.fc6(h5))
            output = self.fcOut(h4)
            output = self.sigmoid(output)
            return output
        
class NeuralNetworkSimpleV128(torch.nn.Module):
        def __init__(self, input_size,nClasses):
            super(NeuralNetworkSimpleV128, self).__init__()
            self.input_size = input_size
            self.h1Size  = 512
            self.h2Size  = 512
            self.h3Size  = 256
            self.h4Size  = 128
#            self.h5Size  = 512
#            self.h6Size  = 512
            self.outSize  = nClasses
            self.fc1 = torch.nn.Linear(self.input_size, self.h1Size)
            self.fc2 = torch.nn.Linear(self.h1Size, self.h2Size)
            self.fc3 = torch.nn.Linear(self.h2Size, self.h3Size)
            self.fc4 = torch.nn.Linear(self.h3Size, self.h4Size)
#            self.fc5 = torch.nn.Linear(self.h4Size, self.outSize)
#            self.fc6 = torch.nn.Linear(self.h5Size, self.h6Size)
            self.fcOut = torch.nn.Linear(self.h4Size, self.outSize)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            h3 = self.relu(self.fc3(h2))
            h4 = self.relu(self.fc4(h3))
#            h5 = self.relu(self.fc5(h4))
#            h6 = self.relu(self.fc6(h5))
            output = self.fcOut(h4)
            output = self.sigmoid(output)
            return output


def getXYSetClef(dataList,featureType,masterClassOrder,dataName,classIdx = 0,imgIdIdx = 1):
    X, Y = [],[]
    if(featureType == "CLIP-I" and dataName =="clef_ip"):
            dataDict = pickle.load(open(baseDir+os.sep+"clef_ip_clean_v2_all_clipFeaturesImage_2712221.p","rb"))
    elif(featureType == "MORRIS-I" and dataName =="clef_ip"):
            dataDict = pickle.load(open(baseDir+os.sep+"clef_ip_clean_v1_imageMobileNetV2DocFigureMorrisEmbeddingsDict_2411221.p","rb"))
    elif(featureType == "MOBILENET-I" and dataName =="clef_ip"):
            dataDict = pickle.load(open(baseDir+os.sep+"clef_ip_clean_v1_imageMobileNetV2ImageNetEmbeddingsDict_2411221.p","rb"))
    for rec in dataList:
        y = np.zeros(len(masterClassOrder))
        y[masterClassOrder.index(rec[classIdx])]= 1
        #imgNm = rec[0].split(".")[0].replace("_","-")
        imgNm = rec[imgIdIdx]
        imgNm = rec[imgIdIdx].split(".")[0]
        if(imgNm in dataDict):
            x = dataDict[imgNm]#.mean(axis=0)
            X.append(np.squeeze(x))
            Y.append(y)
#            if(len(Y)%2000==0):
#                print("data added: ",len(Y))
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))

def getXYSetCustomfile(dataList,masterClassOrder,fileName,classIdx = 0,imgIdIdx = 1):
    X, Y = [],[]
    dataDict = pickle.load(open(baseDir+os.sep+fileName,"rb"))
    for rec in dataList:
        y = np.zeros(len(masterClassOrder))
        y[masterClassOrder.index(rec[classIdx])]= 1
        #imgNm = rec[0].split(".")[0].replace("_","-")
        imgNm = rec[imgIdIdx]
        imgNm = rec[imgIdIdx].split(".")[0]
        if(imgNm in dataDict):
            x = dataDict[imgNm]#.mean(axis=0)
            X.append(np.squeeze(x))
            Y.append(y)
#            if(len(Y)%2000==0):
#                print("data added: ",len(Y))
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))

def getXYSet(dataList,dataDict,featureType,imageCliptextEmbeddings,imageBerttextEmbeddings,all_clipFeaturesImage,masterClassOrder):
    X, Y = [],[]
    for rec in dataList:
        y = np.zeros(len(masterClassOrder))
        y[masterClassOrder.index(rec[1])]= 1
        #imgNm = rec[0].split(".")[0].replace("_","-")
        imgNm = rec[0]
        imgNm = rec[0].split(".")[0]
        if(featureType == "CLIP-T"):
            if(imgNm in imageCliptextEmbeddings):
                x = imageCliptextEmbeddings[imgNm].mean(axis=0)
                X.append(x)
                Y.append(y)
        elif(featureType == "CLIP-I"):
            if(imgNm in all_clipFeaturesImage):
#                shape2Ch = all_clipFeaturesImage[imgNm].shape
                x = all_clipFeaturesImage[imgNm].reshape(-1)#.reshape((shape2Ch[0],shape2Ch[-1]))
                X.append(x)
                Y.append(y)
        elif(featureType == "BERT-T"):
            if(imgNm in imageBerttextEmbeddings):
                x = np.squeeze(imageBerttextEmbeddings[imgNm]).mean(axis=0).astype(np.float)
#                print(featureType+" x.shape: ",x.shape)
                X.append(x)
                Y.append(y)
        elif(featureType == "CLIP-BERT-TT"):
            if(imgNm in imageCliptextEmbeddings and imgNm in imageBerttextEmbeddings):
                f1 = imageCliptextEmbeddings[imgNm].mean(axis=0)
                f2 = np.squeeze(imageBerttextEmbeddings[imgNm]).mean(axis=0).astype(np.float)
                X.append(np.hstack([f1,f2]))
                Y.append(y)
        elif(featureType == "CLIP-BERT-IT"):
#            dataDict = pickle.load(open(baseDir+os.sep+"allwithImageTypeDataDictWithData.p","rb"))
            #dataDictTestB = pickle.load(open(baseDir+os.sep+"testB_clipFeaturesImage.p","rb"))
#            dataDictTestA = pickle.load(open(baseDir+os.sep+"testA_clipFeaturesImage.p","rb"))
            if(imgNm in all_clipFeaturesImage and imgNm in imageBerttextEmbeddings):
                f1 ,f2= [], []
                f1 = all_clipFeaturesImage[imgNm].reshape(-1)
#                if(imgNm in dataDict):
#                    f1 = dataDict[imgNm]
#                if(imgNm in dataDictTestA):
#                    f1 = dataDictTestA[imgNm][0]
                f2 = np.squeeze(imageBerttextEmbeddings[imgNm]).mean(axis=0).astype(np.float)
                X.append(np.hstack([f1,f2]))
                Y.append(y)
        elif(featureType == "CLIP-CLIP-IT"):
#            dataDict = pickle.load(open(baseDir+os.sep+"allwithImageTypeDataDictWithData.p","rb"))
            #dataDictTestB = pickle.load(open(baseDir+os.sep+"testB_clipFeaturesImage.p","rb"))
#            dataDictTestA = pickle.load(open(baseDir+os.sep+"testA_clipFeaturesImage.p","rb"))
            f1 ,f2= [], []
            if(imgNm in all_clipFeaturesImage and imgNm in imageCliptextEmbeddings):
                f1 = all_clipFeaturesImage[imgNm].reshape(-1)
#                if(imgNm in dataDict):
#                    f1 = dataDict[imgNm]
#                if(imgNm in dataDictTestA):
#                    f1 = dataDictTestA[imgNm][0]
                f2 = imageCliptextEmbeddings[imgNm].mean(axis=0)
                X.append(np.hstack([f1,f2]))
                Y.append(y)
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))
import seaborn as sns
def plotConfMatrix(cf_matrix,masterClassOrder,fileN,dtLbs,show=False):
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='g')
    ax.set_title('Confusion Matrix with labels\n\n');
    ax.set_xlabel(dtLbs[0])
    ax.set_ylabel(dtLbs[1]);
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(masterClassOrder)
    ax.yaxis.set_ticklabels(masterClassOrder)
    ## Display the visualization of the Confusion Matrix.
    plt.savefig(fileN+"_confusionMatrix.jpg",dpi=200, bbox_inches='tight')
    plt.clf()
    if(show):
        plt.show()
    
import matplotlib.pyplot as plt
import datetime
def getDtStamp():
    nowis = datetime.datetime.today()
    strDTStamp = str(nowis.day).zfill(2)+str(nowis.month).zfill(2)+str(nowis.year).zfill(2)
    timeStamp = str(nowis.hour).zfill(2)+str(nowis.minute).zfill(2)+str(nowis.second).zfill(2)
    finalStamp = strDTStamp+"_"+timeStamp
    return finalStamp
def convertToMaxIdx(preds):
    labels = []
    for x in preds:
        labels.append(np.argmax(x))
    return labels

def getWithInitLetters(dataLs,n):
    return [x[:n] for x in dataLs]

from sklearn.metrics import confusion_matrix
import time
strTime = time.time()
import random

dtLbs = ['\n Ground Truth Labels','Predictions']
#masterClassOrder = ['block_diagram', 'circuit_diagram', 'flowchart', 'graph','technical_drawing']
allArgs = getCommandLineArgs()

outFileName = getDtStamp()+"_"+allArgs["--modelName"]+"_"+allArgs["--featureType"]+"_lr_"+allArgs["--lRate"]
baseDir = allArgs["--dataLocation"]
outputLocation = "outDir_"+getDtStamp().split("_")[0] #allArgs["outputLocation"]
lRate = float(allArgs["--lRate"])

trainData = allArgs["--trainData"]
trainSubPatentFigurTypeDatasetFile = baseDir+os.sep+trainData+".csv" #"trainSubPatentFigurTypeDataset.csv"
trainSubPatentFigurTypeDataset= pd.read_csv(trainSubPatentFigurTypeDatasetFile).values
trainSubPatentFigurTypeDataset.shape
validationSubPatentFigurTypeDatasetFile = baseDir+os.sep+allArgs["--valData"]+".csv" #"validationSubPatentFigurTypeDataset.csv"
validationSubPatentFigurTypeDataset = pd.read_csv(validationSubPatentFigurTypeDatasetFile).values
validationSubPatentFigurTypeDataset.shape
trainSubPatentFigurTypeDatasetShuffled = shuffle(trainSubPatentFigurTypeDataset, random_state=0)
trainSubPatentFigurTypeDatasetShuffled.shape

featureType = allArgs["--featureType"]
modelName = allArgs["--modelName"]
valData = allArgs["--valData"]
testData = allArgs["--testData"] #"testBData"
featureType = allArgs["--featureType"] #"CLIP-I"
bestmodelcrit = allArgs["--bestmodelcrit"]
datasetName = allArgs["--datasetname"]

#outFileName = getDtStamp()+"_"+modelName+"_"+featureType+"_lr_"+str(lRate).replace(".","_")+"_testData_"+testData+"_valData_"+valData+"_bestmodelcrit_"+bestmodelcrit+"_epochs_"+str(int(allArgs["--epoch"]))
outFileName = getDtStamp()+"_"+modelName+"_"+featureType+"_lr_"+str(lRate).replace(".","_")+"_tstDt_"+testData+"_valDt_"+valData+"_bstcrt_"+bestmodelcrit+"_eps_"+str(int(allArgs["--epoch"]))


isExist = os.path.exists(outputLocation)
if not isExist:
    os.makedirs(outputLocation)
    
testPatentFigurTypeDatasetFile = baseDir+os.sep+testData+".csv" #"testPatentFigurTypeDataset_v2_relabeled.csv"
testPatentFigurTypeDataset = pd.read_csv(testPatentFigurTypeDatasetFile).values
testPatentFigurTypeDataset.shape
#list(dataDict.keys())[:100] # len(list(dataDict.keys()))



imgNm = trainSubPatentFigurTypeDatasetShuffled[10][0]
#imgNmPure = imgNm.split(".")[0]
#smpDt = all_clipFeaturesImage[listAll_clipFeaturesImage[0]]
#smpDt.shape
clasIndex = 1
imgIndex = 0
masterClassOrder = list(np.sort(list(set(testPatentFigurTypeDataset[:,clasIndex]))))
print("masterClassOrder:",masterClassOrder)
#for ky in dataDict: #EP10168987NWB1-i
#    if('EP1016' in ky):
#        print("have : ",ky)
if(datasetName == "expresvip"):
    dateTag = "_101022"
    dataDict = pickle.load(open(baseDir+os.sep+"allwithImageTypeDataDictWithData.p","rb"))
    imageCliptextEmbeddings = pickle.load(open(baseDir+os.sep+"imageCliptextEmbeddings"+dateTag+".p","rb"))
    imageBerttextEmbeddings = pickle.load(open(baseDir+os.sep+"imageBerttextEmbeddings"+dateTag+".p","rb"))
    all_clipFeaturesImage = pickle.load(open(baseDir+os.sep+"all_clipFeaturesImage_081022.p","rb"))
    listAll_clipFeaturesImage = list(all_clipFeaturesImage.keys())

    x_train, y_train = getXYSet(trainSubPatentFigurTypeDatasetShuffled,dataDict,featureType,imageCliptextEmbeddings,imageBerttextEmbeddings,all_clipFeaturesImage,masterClassOrder)
    x_val, y_val= getXYSet(validationSubPatentFigurTypeDataset,dataDict,featureType,imageCliptextEmbeddings,imageBerttextEmbeddings,all_clipFeaturesImage,masterClassOrder)
    x_test, y_test= getXYSet(testPatentFigurTypeDataset,dataDict,featureType,imageCliptextEmbeddings,imageBerttextEmbeddings,all_clipFeaturesImage,masterClassOrder)

print("datasetName:",datasetName)
if(datasetName == "clef_ip"):
    print("compiling train test batches..!!!")
    x_train, y_train = getXYSetClef(trainSubPatentFigurTypeDatasetShuffled,featureType,masterClassOrder,dataName=datasetName,classIdx = clasIndex,imgIdIdx = imgIndex)
    x_val, y_val = getXYSetClef(validationSubPatentFigurTypeDataset,featureType,masterClassOrder,dataName=datasetName,classIdx = clasIndex,imgIdIdx = imgIndex)
    x_test, y_test = getXYSetClef(testPatentFigurTypeDataset,featureType,masterClassOrder,dataName=datasetName,classIdx = clasIndex,imgIdIdx = imgIndex)

if(datasetName == "ustpo_perspective"):
    x_train, y_train = getXYSetCustomfile(trainSubPatentFigurTypeDatasetShuffled,masterClassOrder,"uspto_perspective_v1_all_clipFeaturesImage_1601231.p",classIdx = clasIndex,imgIdIdx = imgIndex)
    x_val, y_val = getXYSetCustomfile(validationSubPatentFigurTypeDataset,masterClassOrder,"uspto_perspective_v1_all_clipFeaturesImage_1601231.p",classIdx = clasIndex,imgIdIdx = imgIndex)
    x_test, y_test = getXYSetCustomfile(testPatentFigurTypeDataset,masterClassOrder,"uspto_perspective_v1_all_clipFeaturesImage_1601231.p",classIdx = clasIndex,imgIdIdx = imgIndex)

print("----------------------------------------")


maxAccGlob = 0.25
maxAccTesty_yHat_Glob = [[],[]]
maxAccPerClassGlob = []
showAt = 1
nRuns = 10
maxPairs = []
maxAccTest = 0.25
model = []
for r in range(nRuns):
    thisSeed = random.randint(1,100000)
    random.seed(thisSeed)
    torch.random.manual_seed(thisSeed)
    logFileName = outputLocation+os.sep+outFileName+"_logs_"+str(thisSeed)
    logFile = open(logFileName+".txt","w")
    #if(os.path.exists(logFileName)):
        
    
    print("allArgs: ",allArgs)
#    print("len all_clipFeaturesImage: ",len(listAll_clipFeaturesImage))
    print("thisSeed: ",thisSeed)
    
    logFile.write("allArgs: "+str(allArgs)+"\n")
#    logFile.write("all_clipFeaturesImage: "+str(len(listAll_clipFeaturesImage))+"\n")
    logFile.write("thisSeed: "+str(thisSeed)+"\n")
    
    logFile.write("x_train shape: "+str(x_train.shape)+"\n")
    logFile.write("y_train shape: "+str(y_train.shape)+"\n")
    logFile.write("x_val shape: "+str(x_val.shape)+"\n")
    logFile.write("y_val shape: "+str(y_val.shape)+"\n")
    logFile.write("x_test shape: "+str(x_test.shape)+"\n")
    logFile.write("y_test shape: "+str(y_test.shape)+"\n")
    inputSize = x_train.shape[1]
    print("inputSize or shape: ",inputSize)
    logFile.write("inputSize or shape:  "+str(inputSize)+"\n")
    if(modelName == "CLNN512_funnel"): # difference is both 512 are neuron and lauer structure, see above
        model = NeuralNetworkSimpleV512_funnel(inputSize,nClasses=len(masterClassOrder))
    elif(modelName == "CLNN512"):
        model = NeuralNetworkSimpleV512(inputSize,nClasses=len(masterClassOrder))
    elif(modelName == "CLNN256"):
        model = NeuralNetworkSimpleV256(inputSize,nClasses=len(masterClassOrder))
    elif(modelName == "CLNN128"):
        model = NeuralNetworkSimpleV128(inputSize,nClasses=len(masterClassOrder))
    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lRate)
    
    
    model.eval()
    y_pred = model(x_test)
    before_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss before training' , before_train.item())
    
    lossTrainLs = []
    lossValLs = []
    testAccLs = []
    testAccPerClassLs = []
    valAccLs = []
    #minTestAcc = 0
    epochs = int(allArgs["--epoch"])
    stopingEta = 0.00000001 # 10^(-8)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        yPredTrain = model(x_train)
        # Compute Loss
        lossTrain = criterion(yPredTrain.squeeze(), y_train)
        yPredVal = model(x_val)
        # Compute Loss
        lossVal = criterion(yPredVal.squeeze(), y_val)
        # Backward pass
        lossTrain.backward()
        optimizer.step()
        lossTrainLs.append(lossTrain.item())
        lossValLs.append(lossVal.item())
        model.eval()
        yPredTest = model(x_test)
        accPerClassVal = getAccuracyMx(y_val.detach().numpy(),yPredVal.detach().numpy(),len(masterClassOrder))
        accVal = round(np.mean(accPerClassVal),3)
        valAccLs.append(accVal)
        accPerClassTest = getAccuracyMx(y_test.detach().numpy(),yPredTest.detach().numpy(),len(masterClassOrder)) 
        accTest = round(np.mean(accPerClassTest),3)
        testAccLs.append(accTest)
        testAccPerClassLs.append(accPerClassTest)
        
        if(accVal>maxAccGlob):
            maxAccGlob = accVal
            maxAccTesty_yHat_Glob = [y_test.detach().numpy(), yPredVal.detach().numpy()]
            maxAccPerClassGlob = accPerClassVal
            outDumpName = outputLocation+os.sep+modelName+"_"+datasetName+"_epoch_"+str(epochs)+"_"+str(thisSeed)
            if not os.path.exists(outputLocation):
                os.makedirs(outputLocation)
            PATH = outDumpName+"_"+str(epoch)+".pt"
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': lossValLs[-1],
                    'train_loss': lossTrainLs[-1],
                    }, PATH, _use_new_zipfile_serialization= False)

            if(epoch%showAt==0):
                print('R: {} Epoch {}: train loss: {}: val loss: {} test Acc: {} : {} val Acc: {} : {}'.format(r,epoch, lossTrain.item(), lossVal.item(),testData,accTest,valData,accVal))
            logFile.write('R: {} Epoch {}: train loss: {}: val loss: {} test Acc: {} : {} val Acc: {} : {}'.format(r,epoch, lossTrain.item(), lossVal.item(),testData,accTest,valData,accVal)+"\n")
            if(epoch>1 and abs(lossValLs[-1]-lossValLs[-2])<stopingEta):
                break
        
            model.eval()
            yPredTest = model(x_test)
            afterTrainTestLoss = criterion(yPredTest.squeeze(), y_test) 
            #print('Test loss after Training' , afterTrainTestLoss.item())
            
            endTime = time.time()
            timeDiff = endTime-strTime
            totalTimeTaken = [timeDiff/3600,timeDiff/60,timeDiff]
            print("total time taken: ",totalTimeTaken)
            logFile.write("total time taken: "+str(totalTimeTaken))
            
            argMaxTest = np.argmax(testAccLs)
            mxAccTest = testAccLs[argMaxTest]
            argMaxVal = np.argmax(valAccLs)
            maxTestAccPerClass = testAccPerClassLs[argMaxTest]
            
            print("outFileName: ",outFileName," : ",testData+"max accTest data: ",testAccLs[argMaxTest],"maxTestAccPerClass:",maxTestAccPerClass,"valAccorTest: ",valAccLs[argMaxTest],valData+"max accVal data: ",valAccLs[argMaxVal]," testAccorVal:",testAccLs[argMaxVal])
            outDictAllVals = {"outFileName":outFileName,testData+"max accTest data":testAccLs[argMaxTest],"maxTestAccPerClass":maxTestAccPerClass,"valAccorTest":valAccLs[argMaxTest],valData+"max accVal data":valAccLs[argMaxVal],"testAccorVal":testAccLs[argMaxVal]}
            maxPairs.append(outDictAllVals)
            logFile.write(str(maxPairs)+"\n")
            print("thisSeed: ",thisSeed)
            logFile.write("thisSeed: "+str(thisSeed))
            
            
            outDict = {"valAccLs":valAccLs,"testAccPerClassLs":testAccPerClassLs,"testAccLs":testAccLs,"totalTimeTaken":totalTimeTaken,"lossTrainLs":lossTrainLs,"lossValLs":lossValLs,"afterTrainTestLoss":afterTrainTestLoss.item(),"testY":y_test.detach().numpy(),"testYhat":yPredTest.squeeze().detach().numpy()}
            print("cwd: ",os.getcwd())
        #    outDumpName = outputLocation+os.sep+outFileName+"_epoch_"+str(epochs)+"_"+testData+"_"+str(thisSeed)
        #    pickle.dump(outDict,open("."+os.sep+outDumpName.replace("-","_")+"_dump.p","wb"))
            pickle.dump(outDict,open(outDumpName+"_"+str(epoch)+"_dump.p","wb"))
            
            fullOutName = outFileName+"_epoch_"+str(epochs)+"_"+testData+"_"+str(thisSeed)
            yLabel = "loss Binary Cross Entropy"
            xLabel = "Epoch number"
            title = "Training and validation loss for all epochs"
            labelLs = ["train","val"]
            dataRows = [lossTrainLs,lossValLs]
            for j in range(len(dataRows)):
                plt.plot(dataRows[j],label=labelLs[j])
            plt.ylabel(yLabel)
            plt.xlabel(xLabel)
            plt.legend(loc="upper left", bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
            plt.title(title)
            plt.savefig(outputLocation+os.sep+outFileName+"_loss"+".jpg",dpi=200, bbox_inches='tight')
        #    plt.savefig(fullOutName+"_loss"+".jpg",dpi=200, bbox_inches='tight')
            plt.clf()
        #    plt.show()
            
            
            yLabel = "Accuracy"
            xLabel = "Epoch number"
            title = "Accuracy for all epochs"
            labelLs = ["test"]
            dataRows = [testAccLs]
            for j in range(len(dataRows)):
                plt.plot(dataRows[j],label=labelLs[j])
            plt.ylabel(yLabel)
            plt.xlabel(xLabel)
            plt.legend(loc="upper left", bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
            plt.title(title)
            plt.savefig(outputLocation+os.sep+outFileName+"_accuracy"+".jpg",dpi=200, bbox_inches='tight')
            plt.clf()
            
            cf_matrix = confusion_matrix(convertToMaxIdx(maxAccTesty_yHat_Glob[0]), convertToMaxIdx(maxAccTesty_yHat_Glob[1])) #,labels=masterClassOrder
            dtLbs = ['\n Predictions','\n Ground Truth Labels']
            masterClassOrderShort = getWithInitLetters(masterClassOrder, n=6)
            plotConfMatrix(cf_matrix,masterClassOrderShort,outputLocation+os.sep+"_"+outFileName,dtLbs,show=False)
            cf_matrixPerc = cf_matrix
            for i in range(len(cf_matrixPerc)):
                cf_matrixPerc[i] = ((cf_matrixPerc[i]/sum(cf_matrixPerc[i]))*100).round(decimals=0)
            # -- psercentage confusion matrix 
            plotConfMatrix(cf_matrixPerc,masterClassOrderShort,"plots"+os.sep+"_"+outFileName+"_perc",dtLbs,show=False)
    #plt.show()
    logFile.close()
print("-----------------------------------------")
print("maxPairs:",maxPairs)
print("-----------------------------------------")
open(outFileName+"_epoch_"+str(epochs)+".txt","w").write(str(maxPairs))

print("maxAccGlob: ",maxAccGlob)
print("maxAccPerClassGlob: ",maxAccPerClassGlob)