from Infor import Infor
class Data:
#初始化输入，权值，标签，每层的误差
	def	__init__(self):
		self.input_xs = [0 for i in range(784)]
		self.weight_input_hidden = [[] for j in range(300)]
		self.output_hidden = [0 for k in range(300)]
		self.weight_hidden_output = [[] for l in range(10)]
		self.output_ys = [0 for i in range(10)]
		self.labels = [0 for i in range(10)]		
		self.error_hidden = [0 for i in range(300)]
		self.error_output = [0 for i in range(10)]		
		self.input_b = [0 for i in range(300)]
		self.hidden_b = [0 for i in range(10)]

	def	_set_input_b(self,input_b):
		self.input_b = input_b
	def	_get_input_b(self):
		return self.input_b

	def	_set_hidden_b(self,hidden_b):
		self.hidden_b = hidden_b
	def	_get_hidden_b(self):
		return self.hidden_b

	def	_set_input_xs(self,input_xs):
		self.input_xs = input_xs
	def	_get_input_xs(self):
		return self.input_xs

	def	_set_weight_input_hidden(self,weight_input_hidden):
		self.weight_input_hidden = weight_input_hidden
	def	_get_weight_input_hidden(self):
		return self.weight_input_hidden

	def	_set_output_hidden(self,output_hidden):
		self.output_hidden = output_hidden
	def	_get_output_hidden(self):
		return self.output_hidden

	def	_set_weight_hidden_output(self,weight_hidden_output):
		self.weight_hidden_output = weight_hidden_output
	def	_get_weight_hidden_output(self):
		return self.weight_hidden_output

	def	_set_output_ys(self,output_ys):
		self.output_ys = output_ys
	def	_get_output_ys(self):
		return self.output_ys
	
	def	_set_labels(self,labels):
		self.labels = labels
	def	_get_labels(self):
		return self.labels

	def	_set_error_hidden(self,error_hidden):
		self.error_hidden = error_hidden
	def	_get_error_hidden(self):
		return self.error_hidden

	def	_set_error_output(self,error_output):
		self.error_output = error_output
	def	_get_error_output(self):
		return self.error_output


