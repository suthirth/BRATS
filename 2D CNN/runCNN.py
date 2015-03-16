import os
from time import gmtime, strftime

import numpy as np
import cPickle

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from classesCNN import ConvPoolLayer, HiddenLayer, LogisticRegression

import readPatMS as ReadPat

#####################
#Training Parameters

learning_rate=0.0001
n_epochs=200
nkerns=[20, 30, 50, 3] 
num_patches=500

#####################
#Load Datasets

num_channels = 4
img_shape = [171,216,171]
num_patients = 4
Tstamps = [4,4,5,4]

valid_pat_num = 5
valid_Tstamps = 4

test_pat_num = 1
test_Tstamp = 1

tr_data = numpy.zeros(Tstamps[0],num_channels,img_shape[0],img_shape[1],img_shape[2])
tr_truth = numpy.zeros(Tstamps[0],img_shape[0],img_shape[1],img_shape[2])
for t in range(0,Tstamps[0]):
    pat = ReadPat.new(1,t+1)
    tr_data[t,:,:,:,:] = pat.data
    tr_truth[t,:,:,:] = pat.truth
train_data = theano.shared(numpy.asarray(tr_data,dtype = theano.config.floatX),borrow = True)
train_truth = theano.shared(numpy.asarray(tr_truth,dtype = 'int32'),borrow = True)

v_data = numpy.zeros(valid_Tstamps,num_channels,img_shape[0],img_shape[1],img_shape[2])
v_truth = numpy.zeros(valid_Tstamps,img_shape[0],img_shape[1],img_shape[2])
for t in range(0,valid_Tstamps):
    vpat = ReadPat.new(valid_pat_num,t+1)
    v_data[t,:,:,:,:] = vpat.data
    v_truth[t,:,:,:] = vpat.truth
valid_data = theano.shared(numpy.asarray(v_data,dtype = theano.config.floatX),borrow = True)
valid_truth = theano.shared(numpy.asarray(pat.truth,dtype = 'int32'),borrow = True)

tpat = ReadPat.new(test_pat_num,test_Tstamp)
test_data = theano.shared(numpy.asarray(tpat.data,dtype = theano.config.floatX),borrow = True)
test_truth = theano.shared(numpy.asarray(tpat.truth,dtype = 'int32'),borrow = True)

#Patch Parameters
patch_size = [19,19]

rng = numpy.random.RandomState(23455)


#Define Theano Tensors
off_x = T.lscalar()
off_y = T.lscalar()
time_idx = T.lscalar()
slice_id = T.lscalar()

x = T.ftensor3('x')  
y = T.ivector('y')  

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

#(num_channels,x,y,patches) -> (patches,num_channels,x,y)
layer0_input = x.dimshuffle(3,0,1,2)

layer0 = ConvPoolLayer(
    rng,
    input=layer0_input,
    img_shape=(num_patches, num_channels, 19, 19),
    filter_shape=(nkerns[0], num_channels, 5, 5),
    poolsize=(2, 2)
)

layer1 = ConvPoolLayer(
    rng,
    input=layer0.output,
    img_shape=(num_patches, nkerns[0], 8, 8),
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
    [off_x, off_y, slice_id],
    layer3.y_pred,
    givens={
        x: T.concatenate[test_data[:, 
                        off_x + ix*patch_size[0]: off_x + (ix+1)*patch_size[0],
                        off_y + iy*patch_size[1]: off_y + (iy+1)*patch_size[1],
                        slice_id] for ix,iy in zip(range((off_x + img_shape[0])/ patch_size[0]),range((off_y + img_shape[1])/patch_size[1])), axis = 3],
        y: T.concatenate[test_truth[
                        off_x + ix*patch_size[0] + (patch_size[0]-1)/2,
                        off_y + iy*patch_size[1] + (patch_size[1]-1)/2,
                        slice_id
                        ] for ix,iy in zip(range((off_x + img_shape[0])/ patch_size[0]),
                                        range((off_y + img_shape[1])/patch_size[1])), axis = 1]
    }
)

validate_model = theano.function(
    [time_idx,off_x,off_y,slice_id],
    layer3.errors(y),
    givens={
    x: T.concatenate[valid_data[time_idx, :, 
                        off_x + ix*patch_size[0]: off_x + (ix+1)*patch_size[0],
                        off_y + iy*patch_size[1]: off_y + (iy+1)*patch_size[1],
                        slice_id
                        ] for ix,iy in zip(range((off_x + img_shape[0])/ patch_size[0]),
                                        range((off_y + img_shape[1])/patch_size[1])), axis = 3],
    y: T.concatenate[valid_truth[time_idx,
                        off_x + ix*patch_size[0] + (patch_size[0]-1)/2,
                        off_y + iy*patch_size[1] + (patch_size[1]-1)/2,
                        slice_id
                        ] for ix,iy in zip(range((off_x + img_shape[0])/ patch_size[0]),
                                        range((off_y + img_shape[1])/patch_size[1])), axis = 1]

    }
)

params = layer3.params + layer2.params + layer1.params + layer0.params

grads = T.grad(cost, params)

updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [time_idx,off_x,off_y,slice_id],
    cost,
    updates=updates,
    givens={
        x: T.concatenate[train_data[time_idx, :, 
                        off_x + ix*patch_size[0]: off_x + (ix+1)*patch_size[0],
                        off_y + iy*patch_size[1]: off_y + (iy+1)*patch_size[1],
                        slice_id
                        ] for ix,iy in zip(range((off_x + img_shape[0])/ patch_size[0]),
                                        range((off_y + img_shape[1])/patch_size[1])), axis = 3],
        y: T.concatenate[train_truth[time_idx,
                        off_x + ix*patch_size[0] + (patch_size[0]-1)/2,
                        off_y + iy*patch_size[1] + (patch_size[1]-1)/2,
                        slice_id
                        ] for ix,iy in zip(range((off_x + img_shape[0])/ patch_size[0]),
                                        range((off_y + img_shape[1])/patch_size[1])), axis = 1]
    }
)

# TRAIN MODEL #
###############
print '... training'

patience = 10000  
patience_increase = 2 
improvement_threshold = 0.995 
validation_frequency = 4000
saving_frequncy = 5000

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    iter = 0
    pat_idx = 0
    for pat_idx in xrange(num_patients):
        
        data = numpy.zeros(TStamps[pat_idx],pat.data.shape[0], pat.data.shape[1],pat.data.shape[2],pat.data.shape[3]],dtype = theano.config.floatX)
        truth = numpy.zeros(TStamps[pat_idx],pat.truth.shape[0], pat.truth.shape[1],pat.truth.shape[2]],dtype = 'int32')
        
        for index in xrange(TStamps[pat_idx]):
            pat = ReadPatMS.new(pat_idx+1,index+1)
            data[index,:,:,:,:] = numpy.asarray(pat.data,dtype = theano.config.floatX)
            truth[index,:,:,:] = numpy.asarray(pat.truth,dtype = 'int32')

        train_patches = np.zeros()


        for index in xrange(TStamps[pat_idx]):
            for z in img_shape[2]:
                    for off_x in patch_size[0]:
                        for off_y in patch_size[1]:
                            num_patches = [(off_x + img_shape[0])/ patch_size[0]), (off_y + img_shape[1])/patch_size[1]]
                            train_patches.append([data[index,:,
                                            off_x + ix*patch_size[0]: off_x + (ix+1)*patch_size[0],
                                            off_y + iy*patch_size[1]: off_y + (iy+1)*patch_size[1],
                                            z] for ix, iy, off_x ,off_y, z, t_index in zip(range(num_patches[0]),range(num_patches[1]))])
                            train_trpatches = [data[index,
                                            off_x + ix*patch_size[0] + (patch_size[0]-1)/2,
                                            off_y + iy*patch_size[1] + (patch_size[1]-1)/2,
                                            z] for ix,iy in zip(range(num_patches[0]),range(num_patches[1]))]
        
        train_data.set_value(train_patches)
        train_truth.set_value(train_trpatches)

        t_idx = 0
        for t_idx in Tstamps[pat_idx]:
            print ('Training: epoch: %i, patient: %i, time stamp: %i \n' %(epoch+1,pat_idx+1,t_idx+1))
            
            for z in xrange(img_shape[2]):
                
                for off_x in xrange(patch_size[0]):
                    
                    for off_y in xrange(patch_size[1]):
                        
                        num_patches = [(img_shape[0]-off_x)/patch_size[0],(img_shape[1]-off_y)/patch_size[1]]

                        cost_ij = train_model(t_idx,off_x,off_y,z)

                        if (iter + 1) % validation_frequency == 0:

                            # compute zero-one loss on validation set
                            validation_losses = [validate_model(t,ox,oy,zz) for t in xrange(valid_Tstamps) 
                                                                            for ox in xrange(patch_size[0]) 
                                                                            for oy in xrange(patch_size[1]) 
                                                                            for zz in range(image_size[2])]
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

Prediction = np.zeros(img_shape)
z = 0
for z in xrange(img_shape[2]):
    pred_ij = test_model(t_idx,0,0,z)
    Prediction[:,:,z] = pred_ij.reshape(img_shape[0],img_shape[1])

end_time = time.clock()
print >> sys.stderr, ('Prediction done and it took '+' %.2fm ' % ((end_time - start_time) / 60.))
#output nii file
affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
img = nib.Nifti1Image(Prediction, affine)
img.set_data_dtype(numpy.int32)
nib.save(img,'prediction.nii')