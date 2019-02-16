from numpy import *
import struct
class readFile:
	def	__init__(self):
		self.a = 0

	def	_readImages(self,filename,index):
		binfile = open(filename,'rb')
		buf = binfile.read()
		binfile.close()
		im = struct.unpack_from('>784B',buf,index)
		return im


	def	_readLabels(self,filename,index):
		binfile = open(filename,'rb')
		buf = binfile.read()
		binfile.close()
		label = struct.unpack_from('1B',buf,index)
		num = label[0]
		vector = [0.1 for i in range(10)]
		f = 0
		for f in range(num):
			pass
		vector[f] = 0.9
		return vector
						
		

#r = readFile()
#out = r._readLabels('/home/cuihang/tensorflowtest/MNIST_data/train-labels-idx1-ubyte')
#print (out)
