import os
from time import gmtime, strftime

import numpy as np
import cPickle

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from classesCNN import ConvPoolLayer, HiddenLayer, LogisticRegression

import readPatGlioma as ReadPat

#####################
#Training Parameters

learning_rate=0.1
n_epochs=200
nkerns=[20, 30, 50, 3] 
batch_size=500

#####################
#Load Datasets

num_channels = 4
image_shape = [171,216,171]
num_patients = 5
Tstamps = [4,4,5,4,3]

valid_pat_num = 5
valid_time_idx = 4

test_pat_num = 1
test_time_idx = 1

data = numpy.zeros(Tstamps[0],num_channels,img_shape[0],img_shape[1],img_shape[2]])
truth = numpy.zeros(Tstamps[0],img_shape[0],img_shape[1],img_shape[2]])

for t in range(0,Tstamps[0]):
    pat = ReadPat.new(1,t+1)
    data[t,:,:,:,:] = pat.data
    truth[t,:,:,:] = pat.truth

train_data = theano.shared(numpy.asarray(pat.data,dtype = theano.config.floatX),borrow = True)
train_truth = theano.shared(numpy.asarray(pat.truth,dtype = 'int32'),borrow = True)

vpat = ReadPat.new(valid_pat_num, valid_time_idx)
valid_data = theano.shared(numpy.asarray(vpat.data,dtype = theano.config.floatX),borrow = True)
valid_truth = theano.shared(numpy.asarray(vpat.truth,dtype = 'int32'),borrow = True)

tpat = ReadPat.new(test_pat_num,test_time_idx)
test_data = theano.shared(numpy.asarray(tpat.data,dtype = theano.config.floatX),borrow = True)
test_truth = theano.shared(numpy.asarray(tpat.truth,dtype = 'int32'),borrow = True)

########################

rng = numpy.random.RandomState(23455)

index = T.lscalar()  
x = T.ftensor3('x')  
y = T.imatrix('y')  

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

layer0_input = x.reshape((batch_size, 1, 20, 20))

layer0 = ConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, 20, 20),
    filter_shape=(nkerns[0], 1, 5, 5),
    poolsize=(2, 2)
)

layer1 = ConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 8, 8),
    filter_shape=(nkerns[1], nkerns[0], 5, 5),
    poolsize=(2, 2)
)

layer2_input = layer1.output.flatten(2)

layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 2 * 2,
    n_out=nkerns[2],
    activation=T.tanh
)

layer3 = LogisticRegression(input=layer2.output, n_in=nkerns[2], n_out=nkerns[3])

cost = layer3.negative_log_likelihood(y)

test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

params = layer3.params + layer2.params + layer1.params + layer0.params

grads = T.grad(cost, params)

updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# TRAIN MODEL #
###############
print '... training'

patience = 10000  
patience_increase = 2 
improvement_threshold = 0.995 
validation_frequency = min(n_train_batches, patience / 2)

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        iter = (epoch - 1) * n_train_batches + minibatch_index

        if iter % 100 == 0:
            print 'training @ iter = ', iter
        cost_ij = train_model(minibatch_index)

        if (iter + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [
                    test_model(i)
                    for i in xrange(n_test_batches)
                ]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

        if patience <= iter:
            done_looping = True
            break

end_time = time.clock()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
