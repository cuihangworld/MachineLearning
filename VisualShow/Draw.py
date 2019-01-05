import numpy as np
import matplotlib.pyplot as plt

def _draw2D(DataMat, w, b):
    for i in range(0, len(DataMat)):
        if 1 == DataMat[i][-1]:
            plt.scatter(DataMat[i][0], DataMat[i][1], color='r')
        else:
            plt.scatter(DataMat[i][0], DataMat[i][1], color='g')
    x1 = 0
    y1 = (w[0]*x1 + b)*(-1)/w[1]
    x2 = 10
    y2 = (w[0] * x2 + b) * (-1) / w[1]
    plt.plot([x1, x2], [y1, y2])

    plt.show()