import numpy as np


def _inputTrainData(path, TrainData):
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

def _CalGramMat(TrainDataMat):
    #delete Mat's the last col which is label
    TrainDataMat = np.delete(TrainDataMat, -1, axis = 1)
    TrainDataMatT = np.transpose(TrainDataMat)
    TrainDataMatMul = np.dot(TrainDataMat, TrainDataMatT)

    return TrainDataMatMul

#def _train():

if __name__ == '__main__':
    path = "train.txt"
    TrainData = []
    TrainDataMat = _inputTrainData(path, TrainData)
    TrainDataMatMul = _CalGramMat(TrainDataMat)

    print (TrainDataMat)
    print(TrainDataMatMul)