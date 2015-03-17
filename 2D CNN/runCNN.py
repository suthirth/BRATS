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

from sklearn.feature_extraction import image

import readPatMS

#####################
#Training Parameters

learning_rate=0.0001
n_epochs=200
nkerns=[20, 30, 50, 8] 

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

pat = readPatMS.new(1,1)
num_patches = np.shape(image.extract_patches_2d(pat.data[0,:,:,0],patch_size))[0]

train_patches = np.zeros([num_patches, num_channels, patch_size[0], patch_size[1]])
trpatches_truth = np.zeros([num_patches])

shared_data = theano.shared(numpy.asarray(train_patches,dtype = theano.config.floatX),borrow = True)
shared_truth = theano.shared(numpy.asarray(trpatches_truth,dtype = 'int32'),borrow = True)

rng = numpy.random.RandomState(23455)


#Define Theano Tensors
nz = T.lscalar()
x = T.ftensor4('x')  
y = T.ivector('y')  

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

layer0input = x.dimshuffle(1,0,2,3)
layer0 = ConvPoolLayer(
    rng,
    input=layer0input,
    image_shape=(num_patches, num_channels, 19, 19),
    filter_shape=(nkerns[0], num_channels, 5, 5),
    poolsize=(2, 2)
)

layer1 = ConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(num_patches, nkerns[0], 8, 8),
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

test_model = theano.function([], layer3.y_pred, givens={ x: shared_data})

validate_model = theano.function([], layer3.errors(y), givens={x: shared_data, y: shared_truth})

params = layer3.params + layer2.params + layer1.params + layer0.params

grads = T.grad(cost, params)

updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

train_model = theano.function([], cost, updates=updates, givens={ x: shared_data, y: shared_truth})

# TRAIN MODEL #
###############
print '... training'

patience = 100000  
patience_increase = 2 
improvement_threshold = 0.995 
validation_frequency = 400000
saving_frequency = 500000

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

logcost = []
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    iter = 0
    
    for pat_idx in xrange(num_patients):
        
        for index in xrange(TStamps[pat_idx]):
                        
            pat = readPatMS.new(pat_idx+1,index+1)

            # Number of batches -> 10 slices
            # Each batch -> 32k patches from each slice
            
            z = 0

            while z < 30*int(img_shape[2]/30):

                print "... preparing patches"

                train_patches0 = image.extract_patches_2d(pat.data[0,:,:,z:z+30], patch_size)
                train_patches1 = image.extract_patches_2d(pat.data[1,:,:,z:z+30], patch_size)
                train_patches2 = image.extract_patches_2d(pat.data[2,:,:,z:z+30], patch_size)
                train_patches3 = image.extract_patches_2d(pat.data[3,:,:,z:z+30], patch_size)
                
                truth_patches = image.extract_patches_2d(pat.truth[:,:,z:z+30], patch_size)
                
                patches_truth = np.zeros([num_patches,30])
                
                for iz in xrange(30):
                    for p in xrange(num_patches):
                        patches_truth[p,iz] = truth_patches[p,(patch_size[0]-1)/2, (patch_size[0]-1)/2,iz]
                          
                for nz in range(30):
                    print ('Training: epoch: %i, patient: %i, time stamp: %i, slice: %i\n' %(epoch,pat_idx+1,index+1,z+nz))
                    shared_data.set_value(numpy.asarray([train_patches0[:,:,:,nz],train_patches1[:,:,:,nz],train_patches2[:,:,:,nz],train_patches3[:,:,:,nz]],dtype = theano.config.floatX))
                    shared_truth.set_value(numpy.asarray(patches_truth[:,nz],dtype = 'int32'))
                    cost_ij = train_model()
                
                z = z + 30
                logcost.append(cost_ij)

                if (iter + 1) % validation_frequency == 0:
                    
                    print "... validation"
                    
                    validation_losses = []
                    for vindex in xrange(valid_Tstamps):

                        vpat = readPatMS.new(valid_pat_num,vindex+1)

                        vz = 0

                        while vz < int(img_shape[2]/10):
               
                            train_patches0 = image.extract_patches_2d(vpat.data[0,:,:,z:z+10], patch_size)
                            train_patches1 = image.extract_patches_2d(vpat.data[1,:,:,z:z+10], patch_size)
                            train_patches2 = image.extract_patches_2d(vpat.data[2,:,:,z:z+10], patch_size)
                            train_patches = image.extract_patches_2d(vpat.data[3,:,:,z:z+10], patch_size)
                            
                            truth_patches = image.extract_patches_2d(pat.truth[:,:,z:z+10], patch_size)

                            patches_truth = np.zeros([num_patches,10])
                            
                            for iz in range(10):
                                for p in num_patches:
                                    patches_truth[p,iz] = p[(patch_size[0]-1)/2, (patch_size[0]-1)/2,iz]

                            z = z + 10

                            shared_data.set_value(numpy.asarray([train_patches0,train_patches1,train_patches2,train_patches3],dtype = theano.config.floatX))
                            shared_truth.set_value(numpy.asarray(patches_truth,dtype = 'int32'))

                            validation_losses = validate_model()
                            print np.shape(validation_losses)
                        
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