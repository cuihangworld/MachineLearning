from Data import Data
from Infor import Infor
class IO:
	def	__init__(self,weight_input_hidden,weight_hidden_output,input_b,hidden_b,d = Data()):
		d._set_input_b(input_b)
		d._set_hidden_b(hidden_b)
		d._set_weight_input_hidden(weight_input_hidden)
		d._set_weight_hidden_output(weight_hidden_output)
		d._set_input_b(input_b)
		d._set_hidden_b(hidden_b)
		
	def	_input_xs(self,input_xs,d = Data()):
		d._set_input_xs(input_xs)

	def	_labels(self,labels,d = Data()):
		d._set_labels(labels)
		
	def	_output_ys(self,d = Data()):
		return d._get_output_ys()

#w = [[1,2,3],[4,5,6],[7,8,9],[1,2,3]]
#x = [[1,2,3],[4,5,6],[7,8,9]]
#d = Data()
#i = IO(w,x,d)
#print (d._get_weight_input_hidden())
#print (d._get_weight_hidden_output())


	


	
