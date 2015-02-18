import time

import mha
import numpy
import cPickle as pickle
from PIL import Image

import theano
import theano.tensor as T

from convnet3d import ConvPoolLayer
from mlp import HiddenLayer
from logistic_sgd import *

learning_rate = 0.1
rng = numpy.random.RandomState(23455)

x = T.matrix('x')
y = T.ivector('y')

layer0_input = x.reshape([37,4,31,31,31])
layer0 = ConvPoolLayer(rng,
					   input = layer0_input,
					   filter_shape = (20,4,5,5,5),
					   image_shape = (37,4,31,31,31),
					   pool_size = (2,2,2)
					   )
ignore_border = True
layer1 = ConvPoolLayer(
	rng, 
	layer0.output, 
	filter_shape = (50,20,5,5,5), 
	image_shape = (37,20,13,13,13), 
	pool_size = (2,2,2)
)
layer2_input = layer1.output.flatten(2)
layer2 = HiddenLayer(rng, input = layer2_input, n_in= 50*64, n_out = 500, activation = T.tanh )
layer3 = LogisticRegression(input = layer2.output,n_in = 500,n_out = 3)
cost = layer3.negative_log_likelihood(y)

test_model = theano.function([x,y],layer3.errors(y))
valid_model = theano.function([x,y],layer3.errors(y))

params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost,params)
updates = [(param_i,param_i-learning_rate*grad_i) for param_i,grad_i in zip(params,grads)]
train_model = theano.function([x,y],cost,updates = updates)



flair = mha.new()
t1 = mha.new()
t2 = mha.new()
t1c = mha.new()
truth = mha.new()

flair.read_mha('patient1/BRATS_HG0001_FLAIR.mha')
t1.read_mha('patient1/BRATS_HG0001_T1.mha')
t2.read_mha('patient1/BRATS_HG0001_T2.mha')
t1c.read_mha('patient1/BRATS_HG0001_T1C.mha')
truth.read_mha('patient1/BRATS_HG0001_truth.mha')

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

patience = 10000
patience_increase = 2
imrovement_threshold = 0.995
validation_frequency = 34*48

best_validation_loss = numpy.inf
best_iter = 0
test_score  = 0.
start_time  = time.clock()

epoch = 0
done_looping = False

while(epoch < n_epochs) and (not done_looping):
	epoch = epoch + 1
	for i in x:
		for j in y:

			itr = (epoch-1)*34*48 + (i/4-4)*48 + j 

			if itr % 100 == 0:
				print 'training @ itr = ',itr

			batch = numpy.zeros([37,4,29791],dtype = numpy.int16)
			label = numpy.zeros(37,dtype = numpy.int16)

			for k in z:
				print 'k = ',k
				combine = [flair.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791),
			   			   t1c.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791),
			   			   t1.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791),
		   	   			   t2.data[(i-1)-15:(i-1)+15+1,(j-1)-15:(j-1)+15+1,(k-1)-15:(k-1)+15+1].reshape(29791)]
		   	   	label[k/4-4] = truth.data[i-1,j-1,k-1]
				batch[k/4-4] = numpy.array(combine)
			costij = train_model(batch,label)

			if (itr + 1) % valiidation_frequency == 0:
				# find validation loss

				if this_validation_loss < best_validation_loss:
					if this_validation_loss < best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					best_itr = itr	

					test_losses = [test_model()]
					test_score = numpy.mean(test_losses)
				if patience <= itr :
					done_looping = True
					break
end_time = time.clock()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print >> sys.stderr, ('The code '+' ran for %.2fm ' % ((end_time - start_time) / 60.))










