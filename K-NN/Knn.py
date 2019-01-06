import numpy as np

import sys
sys.path.append('E:\机器视觉\机器学习\MachineLearning\DataFeed')
import InputData

# create kd-tree, the input is x and label y
def _createKdTree(DataXYMat):
    # sort mat depend on the first col
    DataXYMat = DataXYMat[DataXYMat[:,0].argsort()]


if __name__ == '__main__':
    path = "train.txt"
    TrainDataMat = InputData._inputData(path)
    _createKdTree(TrainDataMat)
