import os
import time
from time import gmtime, strftime

import nibabel as nib

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
n_epochs=30
nkerns=[40, 50, 200, 8] 

#####################
#Load Datasets

num_channels = 4
x = (20,160)
y = (17,192)
z = (11,151)
img_shape = [181,217,181]
num_patients = 4
TStamps = [4,4,5,4]

patch_size = [19,19]

valid_pat_num = 5
valid_Tstamps = 4

test_pat_num = 1
test_Tstamp = 1

pat = readPatMS.new(1,1)
num_patches = np.shape(image.extract_patches_2d(pat.data[0,20:160,17:192,0],patch_size))[0]

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

validate_model = theano.function([nz], layer3.errors(y), givens={x: shared_data[:,nz*num_patches:(nz+1)*num_patches,:,:], 
                                                                y: shared_truth[nz*num_patches:(nz+1)*num_patches]})

params = layer3.params + layer2.params + layer1.params + layer0.params

grads = T.grad(cost, params)

updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

train_model = theano.function([nz], cost, updates=updates, givens={ x: shared_data[:,nz*num_patches:(nz+1)*num_patches,:,:], 
                                                                    y: shared_truth[nz*num_patches:(nz+1)*num_patches]})

# TRAIN MODEL #
###############
print '... training'

saving_frequency = 150

start_time = time.clock()

epoch = 0

logcost = []

while (epoch < n_epochs):
    epoch = epoch + 1
    iter = 0
    
    for pat_idx in xrange(num_patients):
        
        for index in xrange(TStamps[pat_idx]):
                        
            #pat = readPatMS.new(pat_idx+1,index+1)
            # Number of batches -> 10 slices
            # Each batch -> 32k patches from each slice
            z = 11

            slice_batch = 35

            while z < 151:

                train_patches = []
                print "... preparing patches"

                for ch in xrange(num_channels):
		    tr_patches = image.extract_patches_2d(pat.data[ch,20:160,17:192,z:z+slice_batch], patch_size)
                    tr_patches = np.transpose(tr_patches,[3,0,1,2])
                    tr_patches = np.reshape(tr_patches,[num_patches*slice_batch,patch_size[0],patch_size[1]])
                    train_patches.append(tr_patches)
		
                truth_patches = image.extract_patches_2d(pat.truth[20:160,17:192,z:z+slice_batch], patch_size)
                truth_patches = np.transpose(truth_patches,[3,0,1,2])
                truth_patches = np.reshape(truth_patches,[num_patches*slice_batch,patch_size[0],patch_size[1]])                
                
                patches_truth = np.zeros([num_patches*slice_batch])
                for i in xrange(num_patches*slice_batch):
                        patches_truth[i] = truth_patches[i,(patch_size[0]-1)/2, (patch_size[0]-1)/2]
                    
                shared_data.set_value(numpy.asarray(train_patches,dtype = theano.config.floatX))
                shared_truth.set_value(numpy.asarray(patches_truth,dtype = 'int32'))
      
                for nz in range(slice_batch):
                    print ('Training: epoch: %i, patient: %i, time stamp: %i, slice: %i\n' %(epoch,pat_idx+1,index+1,z+nz))
                    cost_ij = train_model(nz)
                
                z = z + slice_batch
                logcost.append(cost_ij)
                
                if (iter+1) % saving_frequency == 0:
                    print 'Saving model...'
                    save_file = file('CNNmodel.pkl', 'wb')
                    for i in xrange(len(params)):
                        cPickle.dump(params[i].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)    
                    save_file.close()
                    
                    myfile = open("logcost.txt", "a") 
                    for l in logcost:
                        myfile.write("%f\n"%l)
                    logcost = []

                    pat = readPatMS.new(1,1)
                    pat_data = numpy.asarray(pat.data,dtype = theano.config.floatX)

                    Prediction = np.zeros([122,157,140])

                    for z in xrange(11,151):
                        print 'Predicting... slice:',(60+z)
                        Pred = []
                        tr_patches = np.asarray([image.extract_patches_2d(pat.data[ch,20:160,17:192,z], patch_size) for ch in xrange(num_channels)])
                        shared_data.set_value(numpy.asarray(tr_patches[:,0:num_patches,:,:],dtype = theano.config.floatX))
                        p1 = test_model() 
                        shared_data.set_value(numpy.asarray(tr_patches[:,num_patches:num_patches*2,:,:],dtype = theano.config.floatX))
                        p2 = test_model()
                        Pred = np.append(p1,p2)
                        Prediction[:,:,z] = np.reshape(np.asarray(Pred),[122,157])

                    #output nii file
                    affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
                    img = nib.Nifti1Image(Prediction, affine)
                    img.set_data_dtype(numpy.int32)
                    nib.save(img,'prediction.nii')

                iter = iter + 1

end_time = time.clock()
print 'Optimization complete. Time taken:', (end_time - start_time)/3600
