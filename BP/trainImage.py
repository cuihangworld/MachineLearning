from Data import Data
from IO import IO
from Operation import Operation
from readFile import readFile
import struct


w1 = [[0 for i in range(784)] for j in range(300)]
w2 = [[0 for i in range(300)] for j in range(10)]
inputb = [0 for i in range(300)]
hiddenb = [0 for j in range(10)]


filename_label = '/home/cuihang/tensorflowtest/MNIST_data/train-labels-idx1-ubyte'
filename_image = '/home/cuihang/tensorflowtest/MNIST_data/train-images-idx3-ubyte'
image_lenth = struct.calcsize('784B')
label_lenth = struct.calcsize('1B')
r = readFile()
d = Data()
i = IO(w1,w2,inputb,hiddenb,d)
index_image = 0
index_image += struct.calcsize('>llll')
index_label = 0
index_label += struct.calcsize('>ll')


for j in range(21):
	input_x = r._readImages(filename_image,index_image)
	i._input_xs(input_x,d)
	index_image += image_lenth
	label = r._readLabels(filename_label,index_label)
	i._labels(label,d)	
	index_label += label_lenth
	o = Operation()
	o._create_forward(d)
	o._error_save(d)
	o._create_back(d)

out = d._get_weight_input_hidden()
print (out)
out1 = d._get_weight_hidden_output()
print (out1)

p = d._get_input_b()
print (p)
q = d._get_hidden_b()
print (q)

r = d._get_output_ys()
print (r)

