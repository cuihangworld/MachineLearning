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

#calculate the Gram mat of the input mat
def _CalGramMat(TrainDataMat):
    #delete Mat's the last col which is label
    TrainDataMat = np.delete(TrainDataMat, -1, axis = 1)
    TrainDataMatT = np.transpose(TrainDataMat)
    TrainDataMatMul = np.dot(TrainDataMat, TrainDataMatT)

    return TrainDataMatMul

#判断是否更新权重，1表示更新，0表示不更新, timeSample是样本次数
def _judgeUpdateWeight(TrainDataMat, a, b, timeSample):
    GramMat = _CalGramMat(TrainDataMat)
    sum = 0
    for i in range(0, len(TrainDataMat)):
        sum = sum + a[i]*TrainDataMat[i][2]*GramMat[timeSample][i]

    if sum*TrainDataMat[timeSample][2] > 0:
        return 0
    else:
        return 1
#训练步长
step = 1
#迭代次数
iteration = 6


# 感知器训练，输入为训练集矩阵
def _train(TrainDataMat):
    a = [0 for i in TrainDataMat]
    b = 0
    for i in range(1, iteration):
        for j in range(0, len(TrainDataMat)):
            if _judgeUpdateWeight(TrainDataMat, a, b, j) == 1:
                a[j] = a[j] + step
                b = b + step * TrainDataMat[j][2]

    TrainDataXMat = np.delete(TrainDataMat, -1, axis=1)
    w = [0 for i in TrainDataXMat]
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

    w = [0 for i in TrainDataXMat]
    b = 0
    w,b = _train(TrainDataMat)
    print (w)
    print (b)