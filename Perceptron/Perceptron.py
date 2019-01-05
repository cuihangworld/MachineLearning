import numpy as np
import sys
sys.path.append('E:\机器视觉\机器学习\MachineLearning\VisualShow')
import Draw

#step of train
step = 1
#time of iteration
iteration = 20

# input train data from train.txt or test.txt
def _inputData(path):
    TrainData = []
    with open(path, 'r') as fileReader:
        while True:
            line = fileReader.readline()
            #want to delete the end '\n', but forget
            #if (not line) or (line[0] == '\n'):
            if (not line):
                break
            row = line.split()
            rowInt = [int(i) for i in row]
            TrainData.append(rowInt)
            TrainDataMat = np.array(TrainData)

    return TrainDataMat

#calculate the Gram mat of the input mat
def _CalGramMat(TrainDataMat):
    #delete Mat's the last col which is label
    TrainDataMat = np.delete(TrainDataMat, -1, axis = 1)
    TrainDataMatT = np.transpose(TrainDataMat)
    TrainDataMatMul = np.dot(TrainDataMat, TrainDataMatT)

    return TrainDataMatMul

#judge whether update the weights，1 is yes，0 is no, timeSample is the time of sample
def _judgeUpdateWeight(TrainDataMat, a, b, timeSample):
    GramMat = _CalGramMat(TrainDataMat)
    sum = 0
    for i in range(0, len(TrainDataMat)):
        sum = sum + a[i]*TrainDataMat[i][2]*GramMat[timeSample][i]
    sum = sum + b
    if sum*TrainDataMat[timeSample][2] > 0:
        return 0
    else:
        return 1


# train perceptron，input is Train data mat
def _train(TrainDataMat):
    a = [0 for i in TrainDataMat]
    b = 0
    accuracy = 0
# dual method of percetron training
    for i in range(1, iteration):
        for j in range(0, len(TrainDataMat)):
            if _judgeUpdateWeight(TrainDataMat, a, b, j) == 1:
                a[j] = a[j] + step
                b = b + step * TrainDataMat[j][2]
            else:
                accuracy = accuracy + 1
        print (accuracy/len(TrainDataMat))
        accuracy = 0
    TrainDataXMat = np.delete(TrainDataMat, -1, axis=1)
    w = [0 for i in TrainDataXMat[0]]
    change = 0
    changeMat = [0 for i in TrainDataXMat]
    for i in range(0, len(TrainDataMat)):
        change = a[i] * TrainDataMat[i][2]
        changeMat = change * TrainDataXMat[i]
        for j in range(0, len(TrainDataXMat[0])):
            w[j] = w[j] + changeMat[j]
    return w,b

if __name__ == '__main__':
     path = "train.txt"
     TrainDataMat = _inputData(path)
     TrainDataMatMul = _CalGramMat(TrainDataMat)
     TrainDataXMat = np.delete(TrainDataMat, -1, axis=1)
    #
     w = [0 for i in TrainDataXMat[0]]
     b = 0
     w,b = _train(TrainDataMat)
     print (w)
     print (b)
     Draw._draw2D(TrainDataMat, w, b)