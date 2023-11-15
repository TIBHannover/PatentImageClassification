import numpy as np
import os
from sklearn.utils import shuffle
import pickle
import torch
import torchvision
import time
import argparse
#import model for extraction
#from torchvision.models.feature_extraction import get_graph_node_names
#from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.tensorboard import SummaryWriter
from torchsampler import ImbalancedDatasetSampler
from torch.multiprocessing import Pool, Process, set_start_method

from patentModels import PatnetCNNModel
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import datetime
def getDtStamp():
    nowis = datetime.datetime.today()
    strDTStamp = str(nowis.day).zfill(2)+str(nowis.month).zfill(2)+str(nowis.year).zfill(2)
    timeStamp = str(nowis.hour).zfill(2)+str(nowis.minute).zfill(2)+str(nowis.second).zfill(2)
    finalStamp = strDTStamp+"_"+timeStamp
    return finalStamp

try:
     set_start_method('spawn')
except RuntimeError:
    pass

class ExtractFeaturesClip(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, clipModel, preprocess):
        self.clipModel = clipModel
        self.preprocess = preprocess

    def __call__(self, imgArr):
        image = self.preprocess(imgArr).unsqueeze(0).to(device)
        return self.clipModel.encode_image(image)

class ExtractFeaturesOtherModels(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, modelExtr,imgSize,featureSize):
        self.modelExtr = modelExtr
        self.imgSize = imgSize
        self.featureSize = featureSize
    
    @torch.no_grad()
    def __call__(self, imgArr):
        #out = self.modelExtr(imgArr.reshape((1,3,self.imgSize ,self.imgSize)))
        out = self.modelExtr(imgArr[None])
        #print("---------------------")
        #print("out['avgpool'].unsqueeze(0)",out["avgpool"].reshape((1,self.featureSize).shape)
        #print("---------------------")
        return out["avgpool"].reshape((1,self.featureSize))

def target_to_oh(target):
    NUM_CLASS = 5  # hard code here, can do partial
    one_hot = torch.eye(NUM_CLASS)[target].view(NUM_CLASS)
    return one_hot


def getKeyCount(target,numClass):
    countDict = {}
    for i in range(numClass):
        countDict[i] = 0
    for x in target:
        #countDict[np.argmax(x.cpu()).item()] += 1
        countDict[x.item()] += 1
    return countDict


use_cuda = torch.cuda.is_available()
deviceStr = "cuda" if use_cuda else "cpu"
device = torch.device(deviceStr)
writer = SummaryWriter()

# below data loader also perform adata augmentation with stated methods (flips, rotations, crops .etc)
def getDataLoader(dataDir,imageSize,featureSize,batch_size,kwargs,thisShuffler,thisSampler,augmentdataFlag):
    rotationDegree = 45
    fliProbability = 0.5
    if(augmentdataFlag== "yes"):
        this_dataset = torchvision.datasets.ImageFolder(
            dataDir,
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomApply([
                    torchvision.transforms.RandomHorizontalFlip(p=fliProbability),
                    torchvision.transforms.RandomVerticalFlip(p=fliProbability),
                    torchvision.transforms.RandomRotation(rotationDegree),
                    torchvision.transforms.RandomResizedCrop(imageSize)]),
                    torchvision.transforms.Resize(imageSize),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                #ExtractFeaturesOtherModels(modelExtraxtor,imageSize[0],featureSize),
                ]
            ),
            #target_transform=target_to_oh,
        )
    else:
        this_dataset = torchvision.datasets.ImageFolder(
            dataDir,
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(imageSize),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                #ExtractFeaturesOtherModels(modelExtraxtor,imageSize[0],featureSize),
                ]
            ),
            #target_transform=target_to_oh,
        )
    if(thisSampler):
        dataLoader = torch.utils.data.DataLoader(
        this_dataset, batch_size=batch_size, shuffle=thisShuffler, sampler=thisSampler(this_dataset), **kwargs
        )
    else:
        dataLoader = torch.utils.data.DataLoader(
        this_dataset, batch_size=batch_size, shuffle=False, sampler=torch.utils.data.RandomSampler(this_dataset,replacement=True), **kwargs
        )
    return dataLoader

def getAccuracyMx(Y,Yht,numClass=5):
    correctCounts = np.zeros(numClass)
    totalCounts = np.zeros(numClass)
    lenAll = len(Y)
    for i in range(lenAll):
        if(Y[i]== np.argmax(Yht[i])):
            correctCounts[Y[i]] += 1 
        totalCounts[Y[i]] += 1
    for i in range(len(totalCounts)):
        if(totalCounts[i]==0):
            totalCounts[i]=1
    accPerClass = correctCounts/totalCounts
    return round(np.mean(accPerClass),3),accPerClass

def main():
    strTime = time.time()
    numOfClass = 10
    parser = argparse.ArgumentParser(description="FC model with torch data pipeline")
    parser.add_argument(
        "--traindata", metavar="DIR", default="./train", help="path to train dataset"
    )
    parser.add_argument(
        "--validationdata",
        metavar="DIR",
        default="./val",
        help="path to validation dataset",
    )
    parser.add_argument("--testdata", metavar="DIR", default="./test", help="path to test dataset")
    parser.add_argument("--datestamp", metavar="datetimestamp", default=getDtStamp().split("_")[0], help="date stamp in case want to continue with specific stamp model")
    parser.add_argument("--output", metavar="DIR", default="./output1", help="path to output dumps")
    parser.add_argument("--modelname", metavar="modelname", default="model1", help="model name to save results")
    parser.add_argument("--basemodel", metavar="basemodel", default="resnet50", help="base convolutional model")
    parser.add_argument("--checkpoints", metavar="checkpoints", default="checkpoints4", help="checkpoints to save model")
    parser.add_argument("--featureSize", metavar="featureSize", default=2048, type=int, help="feature output size from basemodel")
    parser.add_argument("--featurelayer", metavar="featurelayer", default=-1, type=int, help="base convolutional model layer of feature")
    parser.add_argument("--imageSize", metavar="imageSize", default=224, type=int, help="input imagesize for basemodel")
    parser.add_argument("--roundN", metavar="roundN", default=4, type=int, help="round number to N after decimal")
    parser.add_argument("--takeNImages", metavar="takeNImages", default=4, type=int, help="takeNImages will be saved as grid in tensorboard")
    
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--workers", default=1, type=int, metavar="N", help="worker for cpu work")
    parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size (default: 256)")
    parser.add_argument("--lr", "--learning-rate", default=0.0001, type=float, metavar="LR", help="initial learning rate")
    parser.add_argument("--seed", type=int, default=45, metavar="S", help="random seed (default: 45)")
    parser.add_argument("--nClasses", type=int, default=10, metavar="10", help="total calsses in dataset)")
    parser.add_argument("--no-cuda", action="store_true", default=True, help="disables CUDA training")
    parser.add_argument("--batchbalance", metavar="batchBalanceFlag", default="yes", help="batch balance")
    parser.add_argument("--augmentdata", metavar="dataAugmentFlag", default="yes", help="data augmentation flag")
    parser.add_argument("--bestmodelcrit", metavar="bestmodelcriteria", default="val", help="this is to select criteria to select best model")
    parser.add_argument("--debugstop", default=-1, type=int, metavar="debugstop", help="this will stop after N batches to see outputs")
    
    args = parser.parse_args()
    #use_cuda = not args.no_cuda and torch.cuda.is_available()
    #args.datestamp = getDtStamp().split("_")[0]
    torch.manual_seed(args.seed)
    kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

    print("args.traindata: ", args.traindata)
    print("args.epochs: ", args.epochs)

    modelType = args.datestamp+"_"+args.basemodel+"_"+args.modelname+"_dataSrc_"+args.testdata.split(os.sep)[-2]+"_batchBalance_"+args.batchbalance+"_augmentData_"+args.augmentdata+"_"+args.testdata.split(os.sep)[-1]+"_"+args.validationdata.split(os.sep)[-1]+ "_epochs_" + str(args.epochs)+ "_lr_" + str(args.lr)+"_seed_"+str(args.seed)
    outFileName = "output_" + str(modelType)
    args.output = "./out_"+args.datestamp
    #args.checkpoints = "./checkpoints_"+args.datestamp
    print("------------------------")
    print("outFileName: ", outFileName)
    print("------------------------")
    traindir = args.traindata # args.lr, args.testdata.split(os.sep)[-1], args.validationdata.split(os.sep)[-1], args.batchbalance, args.augmentdata
    valdir = args.validationdata
    testdir = args.testdata
    roundN = args.roundN
    imageSize = (args.imageSize,args.imageSize)

    if(args.batchbalance=="yes"):
        train_loader = getDataLoader(traindir,imageSize,args.featureSize,args.batch_size,kwargs,thisShuffler=False,thisSampler=ImbalancedDatasetSampler,augmentdataFlag=args.augmentdata)
    else:
        train_loader = getDataLoader(traindir,imageSize,args.featureSize,args.batch_size,kwargs,thisShuffler=True,thisSampler=False,augmentdataFlag=args.augmentdata)
    val_loader = getDataLoader(valdir,imageSize,args.featureSize,args.batch_size,kwargs,thisShuffler=False,thisSampler=False,augmentdataFlag="no")
    test_loader = getDataLoader(testdir,imageSize,args.featureSize,args.batch_size,kwargs,thisShuffler=False,thisSampler=False,augmentdataFlag="no")
    
    model = PatnetCNNModel(classes=args.nClasses,pretrained=True,baseModel=args.basemodel,featureSize=args.featureSize)
    if not deviceStr == "cpu":
        model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    def test_process(test_loader,model):
        model.eval()
        test_targets, test_preds = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_targets += list(target.cpu().numpy())
                test_preds += list(output.cpu().numpy())
        return test_targets, test_preds


    def train_process(epoch, train_loader, val_loader,model,train_loss_ep,val_loss_ep,testAcLs,valAcLs,accPerClassTestLs,args):
        train_loss_ls = []
        batch_idx = 0
        model.train()
        currValLoss = -1
        currTestAcc = -1
        currValAcc = -1
        subTestAcLs = [-1]
        subValAcLs = [-1]
        if(len(val_loss_ep)>0):
            currValLoss = val_loss_ep[-1]
        if(len(testAcLs)>0 and len(valAcLs)>0):
            currTestAcc = testAcLs[-1]
            currValAcc = valAcLs[-1]
            subTestAcLs = testAcLs
            subValAcLs = valAcLs
        for data, target in train_loader:
        #for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #print("target.shape",target.shape)
            #print("output.shape",output.shape)
            #print("---------------------------")
            #print("some train tergets: ",list(target.cpu().detach().numpy())[:5])
            #print("some train output: ",list(output.cpu().detach().numpy())[:5])
            #print("---------------------------")
            loss_train = criterion(output, target)
            loss_train.backward()
            optimizer.step()
            train_loss_ls.append(loss_train.item())
            batch_idx += 1
            print("epoch: ",
                        epoch,
                        " batchId: ",
                        batch_idx,
                        "loss_train: ",
                        round(loss_train.item(), roundN),
                        "loss_val: ",
                        round(currValLoss, roundN),
                        " data:",
                        list(data.shape),
                        " target:",
                        list(target.shape),
                        "targetCount: ",
                        getKeyCount(target,args.nClasses),
                        "currTestAcc",currTestAcc,
                        "maxTestAcc",max(subTestAcLs),
                        "currValAcc",currValAcc,
                        "maxValAcc",max(subValAcLs),"accPerClassTestLs: ",accPerClassTestLs[-1]
                    )
            if(args.debugstop>0 and batch_idx>=args.debugstop):
                break
        model.eval()
        val_loss_ls = []
        val_targets, val_preds = [], []
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_val = criterion(output, target)
            val_loss_ls.append(loss_val.item())
            val_targets += list(target.cpu().detach().numpy())
            val_preds += list(output.cpu().detach().numpy())
        return np.mean(train_loss_ls), np.mean(val_loss_ls), val_targets, val_preds

    
    if(args.takeNImages>0):
        images, labels = next(iter(train_loader))
        # to do convert images back to unnormalized using imageNet values #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        grid = torchvision.utils.make_grid(images[:args.takeNImages])
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)
    

    train_loss_ep = []
    val_loss_ep = []
    val_loss_checkpoint = []
    test_acc_checkpoint = []
    val_acc_checkpoint = []
    accPerClassTestLs = [[0,0,0,0,0]]
    pathLs = []
    testAcLs = []
    valAcLs = []
    allAccTestLs = []
    allAccTestLsClass = []
    statusDictFile = modelType+"_statusDict"
    if(not os.path.exists(args.output)):
        os.makedirs(args.output)
    isExist = os.path.exists(args.output + os.sep +statusDictFile+".txt")
    PATH = ""
    dictVals = {"valAcc":0.0,"testAcc":0.0,"testData":args.testdata.split(os.sep)[-1],"modelPath":"","epoch":0,"epochs":args.epochs,"batchbalance":args.batchbalance,"augmentdata":args.augmentdata,"validationdata":args.validationdata.split(os.sep)[-1],"lr":args.lr}
    epochStr = 0
    isExist = False
    if isExist:
        print("isExist:",str(isExist)," file:",statusDictFile)
        dictVals = eval(open(args.output + os.sep +statusDictFile+".txt","r").read())
        if("modelPath" in dictVals and dictVals["modelPath"] != ""):
            checkpoint = torch.load(dictVals["modelPath"])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochStr = checkpoint['epoch'] +1
            if(args.output + os.sep +statusDictFile+"_"+str(epochStr-1)+".p"):
                statusDict = pickle.load(open(args.output + os.sep +statusDictFile+"_"+str(epochStr-1)+".p","rb"))
                testAcLs = statusDict["testAcLs"]
                valAcLs = statusDict["valAcLs"]
                if("val_loss_ep" in statusDict):
                    val_loss_ep = statusDict["val_loss_ep"]
            print("****************** EXISTING LOADED FROM SAVED ******************")
    #else:
    #    epochStr = 0
    #    # args.lr, args.testdata.split(os.sep)[-1], args.validationdata.split(os.sep)[-1], args.batchbalance, args.augmentdata
    #    #"batchbalance":args.batchbalance,"augmentdata":args.augmentdata,"validationdata":args.validationdata.split(os.sep)[-1],"lr":args.lr}
    #    statusDict = {"valAcc":0.0,"testAcc":0.0,"testData":args.testdata.split(os.sep)[-1],"modelPath":PATH,"epoch":0,"epochs":args.epochs,"batchbalance":args.batchbalance,"augmentdata":args.augmentdata,"validationdata":args.validationdata.split(os.sep)[-1],"lr":args.lr}
    #    strStatusDict = str(statusDict)
    #    open(args.output + os.sep +statusDictFile+".txt","w").write(strStatusDict)
    #    dictVals = eval(open(args.output + os.sep +statusDictFile+".txt","r").read())
    #    print("****************** NEW FIRST TIME SAVED ******************")
    earlyStopingEta = 0.00001
    for epoch in range(epochStr,args.epochs):
        train_loss, val_loss, val_targets, val_preds = train_process(epoch, train_loader, val_loader,model,train_loss_ep,val_loss_ep,testAcLs,valAcLs,accPerClassTestLs,args)
        train_loss_ep.append(train_loss)
        # if(len(train_loss_ep)>2 and train_loss_ep[-1]-train_loss_ep[-2]<=earlyStopingEta):
        #     break
        val_loss_ep.append(val_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        if(epoch%1==0):
            test_targets, test_preds = test_process(test_loader,model)
            testAcc,accPerClassTest = getAccuracyMx(test_targets,test_preds,args.nClasses)
            accPerClassTestLs.append(accPerClassTest)
            print("accPerClassTest: ",list(accPerClassTest))
            print("thisAvg: ",np.mean(list(accPerClassTest)))
            allAccTestLsClass.append(list(accPerClassTest))
            allAccTestLs.append(np.mean(list(accPerClassTest)))
            valAcc,accPerClassVal = getAccuracyMx(val_targets,val_preds,args.nClasses)
            testAcLs.append(testAcc)
            valAcLs.append(valAcc)
            modelCriteriaBool = True
            if(args.bestmodelcrit == "val"):
                modelCriteriaBool = valAcc >= dictVals["valAcc"]
            # modelCriteriaBool = False
            if(modelCriteriaBool):
                PATH = "./"+args.checkpoints+"/model_"+modelType+"_"+str(epoch)+".pt"
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss_ep[-1],
                        'train_loss': train_loss_ep[-1],
                        }, PATH)
                pathLs.append(PATH)
                val_loss_checkpoint.append(val_loss)
                test_acc_checkpoint.append(testAcc)
                val_acc_checkpoint.append(valAcc)
                statusDict = {"valAcc":valAcc,"testAcc":testAcc,"testData":args.testdata.split(os.sep)[-1],"modelPath":PATH,"epoch":epoch,"epochs":args.epochs,"batchbalance":args.batchbalance,"augmentdata":args.augmentdata,"validationdata":args.validationdata.split(os.sep)[-1],"lr":args.lr}
                strStatusDict = str(statusDict)
                open(args.output + os.sep +statusDictFile+".txt","w").write(strStatusDict)
                statusDict["testAcLs"] = testAcLs
                statusDict["valAcLs"] = valAcLs
                statusDict["test_targets"] = test_targets
                statusDict["test_preds"] = test_preds
                statusDict["val_loss_ep"] = val_loss_ep
                pickle.dump(statusDict,open(args.output + os.sep +statusDictFile+"_"+str(epoch)+".p","wb"))
                print("****************** UPDATED MODEL SAVED ******************")
    maxAccIdx = -1
    if(args.bestmodelcrit == "val"):
        maxAccIdx = np.argmax(val_acc_checkpoint)
    PATH = pathLs[maxAccIdx]
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochV = checkpoint['epoch']
    lossV = checkpoint['val_loss']
    test_targets, test_preds = test_process(test_loader,model)


    endTime = time.time()
    timeDiff = endTime - strTime
    totalTimeTaken = [timeDiff / 3600, timeDiff / 60, timeDiff]
    print("total time taken: ", totalTimeTaken)
    argMx = np.argmax(allAccTestLs)
    print("maxx acc test:",allAccTestLs[argMx])
    print("maxx acc test Class:",allAccTestLsClass[argMx])
    outDict = {
        "totalTimeTaken": totalTimeTaken,
        "lossTrainLs": train_loss_ep,
        "lossValLs": val_loss_ep,
        "testY": test_targets,
        "testYhat": test_preds,
    }
    pickle.dump(outDict, open(args.output + os.sep + outFileName  + "_dump.p", "wb"))
    writer.close()
if __name__ == "__main__":
    main()

