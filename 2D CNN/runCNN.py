import os
import time
from time import gmtime, strftime

import numpy as np
import numpy
import cPickle

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from classesCNN import ConvPoolLayer, HiddenLayer, LogisticRegression

import readPatMS

#####################
#Training Parameters

learning_rate=0.0001
n_epochs=200
nkerns=[2, 3, 5, 8] 

#####################
#Load Datasets

num_channels = 4
img_shape = [181,217,181]
num_patients = 4
TStamps = [4,4,5,4]

patch_size = [19,19]

valid_pat_num = 5
valid_Tstamps = 4

test_pat_num = 1
test_Tstamp = 1

num_batches = patch_size[0]*patch_size[1]
num_patches = [int(img_shape[i]/patch_size[i])-1 for i in range(2)]

train_patches = np.zeros([num_batches, num_patches[0]*num_patches[1], num_channels, patch_size[0], patch_size[1]])
trpatches_truth = np.zeros([num_batches, num_patches[0]*num_patches[1]])

shared_data = theano.shared(numpy.asarray(train_patches,dtype = theano.config.floatX),borrow = True)
shared_truth = theano.shared(numpy.asarray(trpatches_truth,dtype = 'int32'),borrow = True)

rng = numpy.random.RandomState(23455)


#Define Theano Tensors
n_batch = T.lscalar()
x = T.ftensor4('x')  
y = T.ivector('y')  

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

layer0 = ConvPoolLayer(
    rng,
    input=x,
    image_shape=(num_patches[0]*num_patches[1], num_channels, 19, 19),
    filter_shape=(nkerns[0], num_channels, 5, 5),
    poolsize=(2, 2)
)

layer1 = ConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(num_patches[0]*num_patches[1], nkerns[0], 8, 8),
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
    [n_batch],
    layer3.y_pred,
    givens={
        x: shared_data[n_batch,:,:,:,:]}
)

validate_model = theano.function(
    [n_batch],
    layer3.errors(y),
    givens={
    x: shared_data[n_batch,:,:,:,:],
    y: shared_truth[n_batch,:] 
    }
)

params = layer3.params + layer2.params + layer1.params + layer0.params

grads = T.grad(cost, params)

updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function([n_batch], cost, updates=updates, givens={ x: shared_data[n_batch,:,:,:,:], y: shared_truth[n_batch,:] })

# TRAIN MODEL #
###############
print '... training'

patience = 10000  
patience_increase = 2 
improvement_threshold = 0.995 
validation_frequency = 4000
saving_frequency = 5000

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    iter = 0
    
    for pat_idx in xrange(num_patients):
        
        for index in xrange(TStamps[pat_idx]):
            
            print ('Training: epoch: %i, patient: %i, time stamp: %i\n' %(epoch,pat_idx+1,index+1))
            
            pat = readPatMS.new(pat_idx+1,index+1)

            pat_data = numpy.asarray(pat.data,dtype = theano.config.floatX)
            pat_truth = numpy.asarray(pat.truth,dtype = 'int32')

            # One batch -> 80+ patches from slice where stride = patch_len
            for z in xrange(img_shape[2]):

                num_batches = patch_size[0]*patch_size[1]
                num_patches = [int(img_shape[i]/patch_size[i])-1 for i in range(2)]
                train_patches = np.zeros([num_batches, num_patches[0]*num_patches[1], num_channels, patch_size[0], patch_size[1]])
                trpatches_truth = np.zeros([num_batches, num_patches[0]*num_patches[1]])
                
                i = 0 
                for off_x in xrange(patch_size[0]):
                    for off_y in xrange(patch_size[1]):
                        train_patches[i,:,:,:,:] = [pat_data[:,
                                        off_x + ix*patch_size[0]: off_x + (ix+1)*patch_size[0],
                                        off_y + iy*patch_size[1]: off_y + (iy+1)*patch_size[1],
                                        z] for ix in range(num_patches[0]) for iy in range(num_patches[1])]
                        trpatches_truth[i,:] = [pat_truth[
                                        off_x + ix*patch_size[0] + (patch_size[0]-1)/2,
                                        off_y + iy*patch_size[1] + (patch_size[1]-1)/2,
                                        z] for ix in range(num_patches[0]) for iy in range(num_patches[1])]
                        i = i+1

                shared_data.set_value(numpy.asarray(train_patches,dtype = theano.config.floatX))
                shared_truth.set_value(numpy.asarray(trpatches_truth,dtype = 'int32'))
                                     
                for n_batch in xrange(patch_size[0]*patch_size[1]):
                    
                    cost_ij = train_model(n_batch)

                    if (iter + 1) % validation_frequency == 0:
                        
                        validation_losses = []
                        for vindex in xrange(valid_Tstamps): 
               
                            vpat = readPatMS.new(valid_pat_num,vindex+1)

                            val_data = numpy.asarray(vpat.data,dtype = theano.config.floatX)
                            val_truth = numpy.asarray(vpat.truth,dtype = 'int32')

                            # compute zero-one loss on validation set
                            for z in xrange(img_shape[2]):

                                num_batches = patch_size[0]*patch_size[1]
                                num_patches = [int(img_shape[i]/patch_size[i])-1 for i in range(2)]
                                val_patches = np.zeros([num_batches, num_patches[0]*num_patches[1], num_channels, patch_size[0], patch_size[1]])
                                val_ptruth = np.zeros([num_batches, num_patches[0]*num_patches[1]])
                            
                                i = 0 
                                for off_x in xrange(patch_size[0]):
                                    for off_y in xrange(patch_size[1]):
                                        val_patches[i,:,:,:,:] = [val_data[:,
                                                        off_x + ix*patch_size[0]: off_x + (ix+1)*patch_size[0],
                                                        off_y + iy*patch_size[1]: off_y + (iy+1)*patch_size[1],
                                                        z] for ix in range(num_patches[0]) for iy in range(num_patches[1])]
                                        val_ptruth[i,:] = [val_truth[
                                                        off_x + ix*patch_size[0] + (patch_size[0]-1)/2,
                                                        off_y + iy*patch_size[1] + (patch_size[1]-1)/2,
                                                        z] for ix in range(num_patches[0]) for iy in range(num_patches[1])]
                                        i = i+1

                                
                                shared_data.set_value(numpy.asarray(val_patches,dtype = theano.config.floatX))
                                shared_truth.set_value(numpy.asarray(val_ptruth,dtype = 'int32'))

                                for n in xrange(num_batches):
                                    validation_losses.append(validate_model(n))
                            
                        this_validation_loss = numpy.mean(validation_losses)
                        print('Epoch %i, validation error %f %%' %
                              (epoch, this_validation_loss * 100.))

                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:

                            #improve patience if loss improvement is good enough
                            if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                                patience = max(patience, iter * patience_increase)

                            # save best validation score and iteration number
                            best_validation_loss = this_validation_loss
                            best_iter = iter

                    if (iter +1) % saving_frequency == 0:
                        print 'Saving model...'
                        save_file = file('CNNmodel.pkl', 'wb')
                        for i in xrange(len(params)):
                            cPickle.dump(params[i].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)    
                        save_file.close()
                        
                        myfile = open("logcost.txt", "a") 
                        for l in logcost:
                            myfile.write("%f\n"%l)
                        logcost = []

                    if patience <= iter:
                        done_looping = True
                        break

                    iter = iter + 1


end_time = time.clock()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))


# test it on the test set
print "Prediction..."
start_time = time.clock()

pat = readPatMS.new(1,1)

pat_data = numpy.asarray(pat.data,dtype = theano.config.floatX)
pat_truth = numpy.asarray(pat.truth,dtype = 'int32')

# One batch -> 80+ patches from slice where stride = patch_len
Prediction = np.zeros(img_shape)
num_batches = patch_size[0]*patch_size[1]
num_patches = [int(img_shape[i]/patch_size[i])-1 for i in range(2)]

for z in xrange(img_shape[2]):

    test_patches = np.zeros([num_batches, num_patches[0]*num_patches[1], num_channels, patch_size[0], patch_size[1]])
    test_pred = np.zeros([num_batches, num_patches[0]*num_patches[1]])
                
    i = 0 
    for off_x in xrange(patch_size[0]):
        for off_y in xrange(patch_size[1]):
            test_patches[i,:,:,:,:] = [pat_data[:,
                            off_x + ix*patch_size[0]: off_x + (ix+1)*patch_size[0],
                            off_y + iy*patch_size[1]: off_y + (iy+1)*patch_size[1],
                            z] for ix in range(num_patches[0]) for iy in range(num_patches[1])]
            i = i+1

    shared_data.set_value(numpy.asarray(test_patches,dtype = theano.config.floatX))

    for off_x in xrange(patch_size[0]):
        for off_y in xrange(patch_size[1]):
            pred = test_model(off_x*off_y + off_y)
            for ix in range(num_patches[0]):
                for iy in range(num_patches[1]):
                    Prediction[off_x + ix*patch_size[0] + (patch_size[0]-1)/2,
                        off_y + iy*patch_size[1] + (patch_size[1]-1)/2,
                        z] = pred[ix*iy + iy] 

end_time = time.clock()
print >> sys.stderr, ('Prediction done and it took '+' %.2fm ' % ((end_time - start_time) / 60.))
#output nii file
affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
img = nib.Nifti1Image(Prediction, affine)
img.set_data_dtype(numpy.int32)
nib.save(img,'prediction.nii')