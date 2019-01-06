import numpy as np

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