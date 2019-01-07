import numpy as np

import sys
sys.path.append('E:\机器视觉\机器学习\MachineLearning\DataFeed')
import InputData

class Node(object):
    def __init__(self, DataMat, Left, Right):
        self.left = Left
        self.right = Right
        self.datamat = DataMat

# create kd-tree, the input is x and label y
def _createKdTree(DataXYMat, SortCol):
    # sort mat depend on the first col
    DataXYMat = DataXYMat[DataXYMat[:,SortCol].argsort()]
    MidPointIndex = int(len(DataXYMat)/2 + 1)
    NULL = [0 for i in DataXYMat[0]]
    node = Node(DataXYMat[MidPointIndex - 1], NULL, NULL)
    leftnodes = DataXYMat[:MidPointIndex-1]
    rightnodes = DataXYMat[MidPointIndex:]
    SortCol = SortCol + 1
    if 1 == len(DataXYMat):
        return node
    elif 2 == len(DataXYMat):
        node.left = _createKdTree(leftnodes, SortCol)
    else:
        node.left = _createKdTree(leftnodes, SortCol)
        node.right = _createKdTree(rightnodes, SortCol)

    return node

if __name__ == '__main__':
    path = "train.txt"
    TrainDataMat = InputData._inputData(path)
    node = _createKdTree(TrainDataMat, 0)

    print (node.right.datamat)