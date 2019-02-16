from numpy import *
from Infor import Infor
#感知器模块
class Perceptron:
	def	__init__(self):
		self.input_xs = []
		self.weights = []
		self.b = 0

#计算每一个感知器的加权和
	def	_sum_weight(self):
		self.input_xs = mat(self.input_xs)
		self.weights = mat(self.weights)
		lists = self.input_xs*self.weights.T+self.b
		return lists[0,0]

#将加权和带入激活函数
	def	_sigmoid(self):
		return 1.0/(1+exp(-self._sum_weight()))

#将权值与输入输入，返回输出
	def	_perceptron(self,input_xs,weights,b):
		self.input_xs = input_xs
		self.weights = weights
		self.b = b
		return self._sigmoid()

#x = [1,2,3]
#w = [4,5,6]
#p = Perceptron()
#print (p._perceptron(x,w,1))


#i = Infor()
#print (i.input_num)
