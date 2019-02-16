'''
import struct
filename = '/home/cuihang/tensorflowtest/MNIST_data/train-labels-idx1-ubyte'
binfile = open(filename,'rb')
buf = binfile.read()
binfile.close()
i = 0
index = 0
index += struct.calcsize('>ll')
for i in range(40): 
	label = struct.unpack_from('1B',buf,index)
	print (label)
	index += struct.calcsize('>1B')
'''

import random

print (random.uniform(-0.1,0.1))
