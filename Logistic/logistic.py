import numpy as np

import sys
sys.path.append('E:\机器视觉\机器学习\MachineLearning\DataFeed')
import InputData

def _sigmoid(OneTrainData):
    colNum = OneTrainData.shape[1] + 1
    w = [0 for i in range(colNum)]
    #OneTrainData第一列加一个1
    data = np.column_stack((OneTrainData, [1]))

    return 1/(1+np.exp(data * w))
def _train(TrainDataMat):
    for i in range(len(TrainDataMat)):
        # 将x与y分离
        x =
        y =
        w = w - rate *

if __name__ == '__main__':
    print (_sigmoid(data))