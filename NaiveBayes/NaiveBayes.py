import numpy as np

import sys
sys.path.append('E:\机器视觉\机器学习\MachineLearning\DataFeed')
import InputData

# create Bayes Classifier, calculate every kind of classes
def _createBayesClassifier(DataXsYMat, NumX):
    calY, clasY, numY = _returnClassNumber(TrainDataMat, 2)
    PY = []
    for i in range(calY):
        PY.append(0)
        PY[i] = numY[i] / (len(DataXsYMat))
    for i in range(NumX):
        calX, clasX, numX = _returnClassNumber(TrainDataMat, i)
        for j in range(len(DataXsYMat)):
            if (PY[0] == DataXsYMat[j][2]) && ():
# the number of classes, which class, number of each class
def _returnClassNumber(DataXsYMat, col):
    clas = []
    clas.append("#")
    cal = 0
    for i in range(len(DataXsYMat)):
        mark = 0
        for j in range(len(clas)):
            if clas[j] == DataXsYMat[i][col]:
                mark = 1
                break
        if 1 == mark:
            pass
        else:
            cal = cal + 1
            clas.append(DataXsYMat[i][col])

    clas.pop(0)
    num = []

    for i in range(len(clas)):
        num.append(0)
        for j in range(len(DataXsYMat)):
            if clas[i] == DataXsYMat[j][col]:
               num[i] = num[i] + 1


    return cal, clas, num
if __name__ == '__main__':
    path = "train.txt"
    TrainDataMat = InputData._inputDataCharacter(path)
    PY = _createBayesClassifier(TrainDataMat)
    print (PY)