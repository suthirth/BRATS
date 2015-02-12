import mha
import numpy
import cPickle as pickle
from PIL import Image

import theano
import theano.Tensor as T




layer2 = HiddenLayer(rng, layer2_input, nin= , nout = , activation = T.tanh )
layer3 = LogisticRegression(input = layer2.output,n_in = 500,n_out = 6)
cost = layer3.negative_log_likelihood(y)

test_model = theano.function([data,label],layer3.errors(y),givens={x:data,y:label})
valid_model = theano.function([data,label],layer3.errors(y),givens={x:data,y:label})

params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost,params)
updates = [(param_i,param_i-learning_rate*grad_i) for param_i,grad_i in zip(params,grads)]
train_model = theano.function([data,label],cost,updates = updates, givens = { x:data, y:label })



flair = mha.new()
t1 = mha.new()
t2 = mha.new()
t1c = mha.new()

flair.read_mha('BRATS_HG0001_FLAIR.mha')
t1.read_mha('BRATS_HG0001_T1.mha')
t2.read_mha('BRATS_HG0001_T2.mha')
t1c.read_mha('BRATS_HG0001_T1C.mha')

"""
176,160,216

16,16 	20,16	24,16	.... 160,16
16,20
.
.
.
.
16,144     	


16,20,....  200 

134217728 elements = 1GB
34*58*38*4*29791 = 55 GB
"""

x = numpy.r_[4:37]*4
y = numpy.r_[4:51]*4
z = numpy.r_[4:41]*4




batch = numpy.zeros([37,4,29791],dtype=numpy.int16)
#for i in x:
#	for j in y:
for k in z:
	combine = [flair.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791),
			   t1c.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791),
			   t1.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791),
		   	   t2.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791)]
	batch[k/4-4] = numpy.array(combine)


#	print i,' ',j




