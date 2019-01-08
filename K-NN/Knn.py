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

# select nearest node, Xn is feature
def _SelectKdTree(NodeTree, DataSelected, Xn, NearestLenth):
    if 0 == NodeTree.left.datamat[-1] and 0 == NodeTree.right.datamat[-1]:
        NodeTreeX = np.delete(NodeTree.datamat, -1, axis = 1)
        #DataSelectedX = np.delete(DataSelected, -1, axis = 1)
        Lenth = np.sqrt(np.sum(np.square(NodeTreeX - DataSelected)))
        if Lenth < NearestLenth:
            NearestLenth = Lenth
            NearestNode  = NodeTree.datamat
        return NearestNode, NearestLenth
    # into left
    elif DataSelected[Xn] < NodeTree.datamat[Xn]:
        # left is null
        if 0 == NodeTree.left.datamat[-1]:
            NodeTreeX = np.delete(NodeTree.datamat, -1, axis=1)
            #DataSelectedX = np.delete(DataSelected, -1, axis=1)
            Lenth = np.sqrt(np.sum(np.square(NodeTreeX - DataSelected)))
            if Lenth < NearestLenth:
                NearestLenth = Lenth
                NearestNode = NodeTree.datamat
            return NearestNode, NearestLenth
        Xn = Xn - 1
        NearestNode, NearestLenth = _SelectKdTree(NodeTree.left, DataSelected, Xn, NearestLenth)
        NodeTreeX = np.delete(NodeTree.datamat, -1, axis=1)
        #DataSelectedX = np.delete(DataSelected, -1, axis=1)
        Lenth = np.sqrt(np.sum(np.square(NodeTreeX - DataSelected)))
        if Lenth < NearestLenth:
            NearestLenth = Lenth
            NearestNode  = NodeTree.datamat
        # using NearestLenth as R draw circle judge whether cover NodeTree's right
        if 0 == NodeTree.right.datamat[-1]:
            # right is null
            return NearestNode, NearestLenth
        else:
            if (DataSelected[Xn] + NearestLenth) < NodeTree[Xn]:
                # uncover
                return NearestNode, NearestLenth

            Xn = Xn - 1
            NearestNode, NearestLenth = _SelectKdTree(NodeTree.right, DataSelected, Xn, NearestLenth)
            return NearestNode, NearestLenth

    elif DataSelected[Xn] > NodeTree.datamat[Xn]:
        # right is null
        if 0 == NodeTree.right.datamat[-1]:
            NodeTreeX = np.delete(NodeTree.datamat, -1, axis=1)
            #DataSelectedX = np.delete(DataSelected, -1, axis=1)
            Lenth = np.sqrt(np.sum(np.square(NodeTreeX - DataSelected)))
            if Lenth < NearestLenth:
                NearestLenth = Lenth
                NearestNode = NodeTree.datamat
            return NearestNode, NearestLenth
        Xn = Xn - 1
        NearestNode, NearestLenth = _SelectKdTree(NodeTree.right, DataSelected, Xn, NearestLenth)
        NodeTreeX = np.delete(NodeTree.datamat, -1, axis=1)
        #DataSelectedX = np.delete(DataSelected, -1, axis=1)
        Lenth = np.sqrt(np.sum(np.square(NodeTreeX - DataSelected)))
        if Lenth < NearestLenth:
            NearestLenth = Lenth
            NearestNode = NodeTree.datamat
        # using NearestLenth as R draw circle judge whether cover NodeTree's right
        if 0 == NodeTree.left.datamat[-1]:
            # right is null
            return NearestNode, NearestLenth
        else:
            if (DataSelected[Xn] + NearestLenth) < NodeTree[Xn]:
                # uncover
                return NearestNode, NearestLenth

            Xn = Xn - 1
            NearestNode, NearestLenth = _SelectKdTree(NodeTree.left, DataSelected, Xn, NearestLenth)
            return NearestNode, NearestLenth




if __name__ == '__main__':
    path = "train.txt"
    TrainDataMat = InputData._inputData(path)

    node = _createKdTree(TrainDataMat, 0)
    DataSelected = np.array([5.5, 3.5])
    Xn = 0
    NearestLenth = 100000

    NearestNode, NearestLenth = _SelectKdTree(, DataSelected, Xn, NearestLenth)
    print (NearestNode)
    print (NearestLenth)