import numpy as np

TrainData = []
TrainDataMat = []
def _inputTrainData(path):
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

def _CalGramMat():
    TrainDataMat = np.mat(TrainData)

#def _train():

if __name__ == '__main__':
    path = "train.txt"
    _inputTrainData(path)
    _CalGramMat()
    print (TrainData)
    print (TrainDataMat)