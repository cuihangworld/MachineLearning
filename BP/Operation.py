from Perceptron import Perceptron
from Data import Data
from IO import IO
from numpy import *
#BP神经网络的构造部分
class Operation:
	def	__init__(self):
		self.p = Perceptron()
		
#构造层，可以是隐藏层，输出层
	def	_create_layer(self,input_xs,weights,bs):
		i = 0
		sum_weights = len(weights)
		output_x = [0 for _ in range(sum_weights)]
		sample = zip(weights,bs)
		for weight,b in sample:
			output_x[i] = self.p._perceptron(input_xs,weight,b)
			i = i + 1
		return output_x

#构造神经网络的正向运算
	def	_create_forward(self,d = Data()):
		input_b = d._get_input_b()
		hidden_b = d._get_hidden_b()
		input_xs = d._get_input_xs()
		weight_input_hidden = d._get_weight_input_hidden()
		output_hidden = self._create_layer(input_xs,weight_input_hidden,input_b)
		d._set_output_hidden(output_hidden)
		weight_hidden_output = d._get_weight_hidden_output()
		output_ys = self._create_layer(output_hidden,weight_hidden_output,hidden_b)
		d._set_output_ys(output_ys)

#输出节点的误差
	def	_error_output(self,output_ys,labels):
		samples = zip(output_ys,labels)
		error_output = [0 for i in range(10)]
		i = 0
		for output_y,label in samples:
			error_output[i]= output_y*(1-output_y)*(label-output_y)
			i = i+1
		return error_output

#隐藏层的节点
	def	_error_hidden(self,weight_hidden_output,error_output,output_hidden):
		weight_hidden_output_mat = mat(weight_hidden_output)
		i = 0
		j = 0
		sum = 0
		error_hidden = [0 for i in range(300)]
		for output_hidden_e in output_hidden:
			s = zip(weight_hidden_output_mat[:,i],error_output)
			for weights,error in s:
				swp = weights.tolist()
				weight = swp[0][0]
				sum = sum + weight*error
			error_hidden[j] = output_hidden_e*(1-output_hidden_e)*sum
			i = i + 1
			j = j + 1
#		error_hidden_list = error_hidden.tolist()				
		return error_hidden

#将各层误差存入数据层
	def	_error_save(self,d = Data()):
#		error_output = self._error_output(d.output_ys,d.labels)
#		error_hidden = self._error_hidden(d.weight_output_hidden,error_output,d.output_hidden)
#		d._set_error_output(error_output)
#		d._set_error_hidden(error_hidden)
		error_output = self._error_output(d._get_output_ys(),d._get_labels())
		d._set_error_output(error_output)
		error_hidden = self._error_hidden(d._get_weight_hidden_output(),error_output,d._get_output_hidden())
		d._set_error_hidden(error_hidden)
	
#更新权值的计算
	def	_up_weight(self,weight,error,out,ratio):
		new_weight = weight + ratio*error*out
		return new_weight

#更新偏置的计算
	def	_up_b(self,b,error,ratio):
		new_b = b + error*ratio
		return new_b


#创建反向传播	
	def	_create_back(self,d = Data()):
		self._error_save(d)
		i = 0
		
		a = d._get_weight_hidden_output()
		b = d._get_error_output()
		hidden_b = d._get_hidden_b()
		get_output_hidden = d._get_output_hidden()
		sample = zip(a,b,hidden_b)
#隐藏层到输出层的权值更新
		weight_hidden_output = [[0 for q in range(300)] for p in range(10)]
		b_hidden = [0 for p in range(10)]
		for weight,error,hidden_bb in sample:
#更新隐藏层的偏置变量
			b_hidden[i] = self._up_b(hidden_bb,error,0.04)
			j = 0
			samples = zip(weight,get_output_hidden)
			for weightt,out in samples:
#				weight_hidden_output[i][j] = self._up_weight(weightt,error,out,0.5)
				outputs = self._up_weight(weightt,error,out,0.04)
				weight_hidden_output[i][j] = outputs
				j = j+1
			i = i+1
		d._set_weight_hidden_output(weight_hidden_output)
		d._set_hidden_b(b_hidden)
		i = 0
#输入层到隐藏层的权值更新
		input_b = d._get_input_b()
		b_input = [0 for p in range(300)]
		up_weight_input_hidden = [[0 for q in range(784)] for p in range(300)]
		error_weight = zip(d._get_error_hidden(),d._get_weight_input_hidden(),input_b)
		for error_hidden,weight_input_hidden,input_b in error_weight:
#更新输入层的偏置变量
			b_input[i] = self._up_b(input_b,error_hidden,0.04)
			j = 0
			input_weight = zip(d._get_input_xs(),weight_input_hidden)
			for input_x,weight_input in input_weight:
				outs = self._up_weight(weight_input,error_hidden,input_x,0.04)
				up_weight_input_hidden[i][j] = outs
				j = j + 1
			i = i + 1	
		d._set_weight_input_hidden(up_weight_input_hidden)
		d._set_input_b(b_input)
		

#w1 = [[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
#w2 = [[1,1,1,1],[1,1,1,1],[1,1,1,1]]
#inputb = [2,2,2,2]
#hiddenb = [2,2,2]
#input_xs = [1,2,3]
#ppp = [0.99966464986953363, 0.99966464986953363, 0.99966464986953363, 0.99966464986953363]
#labels = [4,5,6]








