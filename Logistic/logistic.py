import numpy as np

import sys
sys.path.append('E:\机器视觉\机器学习\MachineLearning\DataFeed')
import InputData

def _sigmoid(OneTrainData, w):
    # #OneTrainData第一列加一个1
    # b = np.ones(1)
    # #拼接矩阵
    # data = np.hstack((b, OneTrainData))
    #转置
    w = w.reshape(w.shape[0], 1)
    data = np.array(OneTrainData)
    w = np.mat(w)
    data = np.mat(data)

    return 1/(1+np.exp(data * w))
def _train(TrainDataMat, numTrain):
    xs = np.delete(TrainDataMat, -1, axis=1)
    ys = np.delete(TrainDataMat, [0,1], axis=1)
    rate = 0.01
    colNum = xs.shape[1] + 1
    wList = [0 for i in range(colNum)]
    cost = [0 for i in range(colNum)]
    cost = np.array(cost)
    w = np.array(wList)
    for j in range(numTrain):
        for i in range(len(TrainDataMat)):
            # 将x与y分离
            x = xs[i]
            b = np.ones(1)
            x = np.hstack((b, x))
            y = ys[i]
            cost = cost + (_sigmoid(x, w) - y) * x



if __name__ == '__main__':
    path = "train.txt"
    TrainDataMat = InputData._inputData(path)
    _train(TrainDataMat, 1)