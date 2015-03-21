"""
x = 16,36,56,76 


x-16 : x
x : x + plen-31 
x + plen-31 : x+plen - 32 +16



"""
import os
import sys
import time
import numpy
import cPickle
from PIL import Image

import nibabel as nib

import theano
import theano.tensor as T

from convnet3d import *
from time import gmtime, strftime

import readPatMS as ReadPat
############################################

resumeTraining = False							# make this true to resume training from saved model	

############################################
# General  Hyper-parameters
learning_rate = 0.01 #e-04 				
n_epochs = 100
saving_frequency = 1000

#network parameters
n_fmaps = (60,60,300,8)		#feature map description :
#n_fmaps[0] - number of feature maps after 1st convolution
#n_fmaps[1] - number of feature maps after 2nd convolution
#n_fmaps[2] - number of output neurons after hidden layer
#n_fmaps[3] - number of output classes							

fmap_sizes = (5,3,2,1)		
# 0th value is fmap_size after 1st convolution and so on
############################################
#"""
#Image details and settings for Multiple Sclerosis
plen = 43
offset = 16
numPred = plen - offset + 1

test_pat_num = 5
test_tstamp = 2

############################################
#Details pertaining to Multiple Sclerosis Dataset
num_channels = 1
num_patients = 4
TStamps = (4,4,5,4)
img_shape = (181,217,181)
num_classes = 8

############################################
#Initial Settings

done_looping = False
rng = numpy.random.RandomState(23455)

if(resumeTraining):
	savedModel = file('CNNmodel','wb')
	genVariables = cPickle.load(savedModel)
	epoch,pat_idx,tim_idx,xi,yi,zi,itr = genVariables
	layer3convW = cPickle.load(savedModel)
	layer3convb = cPickle.load(savedModel)
	layer2convW = cPickle.load(savedModel)
	layer2convb = cPickle.load(savedModel)
	layer1convW = cPickle.load(savedModel)
	layer1convb = cPickle.load(savedModel)
	layer0convW = cPickle.load(savedModel)
	layer0convb = cPickle.load(savedModel)
	
else:
	epoch,pat_idx,tim_idx,xi,yi,zi,itr = [0,0,0,0,0,0,0]
	layer3convW = None
	layer3convb = None
	layer2convW = None
	layer2convb = None
	layer1convW = None
	layer1convb = None
	layer0convW = None
	layer0convb = None
######################################
##Loading Dataset to Shared space

data = numpy.zeros([numpy.max(TStamps),num_channels,img_shape[0],img_shape[1],img_shape[2]])
truth = numpy.zeros([numpy.max(TStamps),img_shape[0],img_shape[1],img_shape[2]])
for ix in xrange(TStamps[0]):
	pat = ReadPat.new(1,ix+1,num_channels)
	data[ix,:,:,:,:] = pat.data
	truth[ix,:,:,:] = pat.truth
train_data = theano.shared(numpy.asarray(data,dtype = theano.config.floatX),borrow = True)
train_truth = theano.shared(numpy.asarray(truth,dtype = 'int32'),borrow = True)

tpat = ReadPat.new(test_pat_num,test_tstamp,num_channels)
test_data = theano.shared(numpy.asarray(tpat.data,dtype = theano.config.floatX),borrow = True)
test_truth = theano.shared(numpy.asarray(tpat.truth,dtype = 'int32'),borrow = True)

shape = tpat.truth.shape
Prediction = numpy.zeros(shape)
shape = numpy.insert(shape,0,num_classes)
PostProbs = numpy.zeros(shape)

######################################
##Build architecture
p_shape = (plen,plen,plen)
#ignore_border = True
idx = T.lscalar()
idy = T.lscalar()
idz = T.lscalar()
time_idx = T.lscalar()

x = T.ftensor4('x')
z = T.itensor3('y')
y = z.reshape([numPred*numPred*numPred,])

layer0_input = x.reshape([1,num_channels,p_shape[0],p_shape[1],p_shape[2]])
layer0conv = ConvLayer(rng,
					   input = layer0_input,
					   filter_shape = (n_fmaps[0],num_channels,fmap_sizes[0],fmap_sizes[0],fmap_sizes[0]),
					   image_shape = (1,num_channels,p_shape[0],p_shape[1],p_shape[2]),
					   W_init = layer0convW,
					   b_init = layer0convb,
					   sparse_count = 0 )
newlen = layer0conv.outputlen

layer0pool = PoolLayer(layer0conv.output,
					   image_shape = (1,n_fmaps[0],newlen[0],newlen[1],newlen[2]),
					   pool_size = (2,2,2),
					   sparse_count = 0)
newlen = layer0pool.outputlen

layer1conv = ConvLayer(rng,
				       layer0pool.output,
				       filter_shape = (n_fmaps[1],n_fmaps[0],fmap_sizes[1],fmap_sizes[1],fmap_sizes[1]),
				       image_shape = (1,n_fmaps[0],newlen[0],newlen[1],newlen[2]), 
					   W_init = layer1convW,
					   b_init = layer1convb,
				       sparse_count = 1 )
newlen = layer1conv.outputlen

layer1pool = PoolLayer(layer1conv.output,
					   image_shape = (1,n_fmaps[1],newlen[0],newlen[1],newlen[2]),
					   pool_size = (2,2,2),
					   sparse_count = 1 )
newlen = layer1pool.outputlen

layer2conv = ConvLayer(rng,
				   input = layer1pool.output,
				   filter_shape = (n_fmaps[2],n_fmaps[1],fmap_sizes[2],fmap_sizes[2],fmap_sizes[2]),
				   image_shape = (1,n_fmaps[1],newlen[0],newlen[1],newlen[2]),
				   W_init = layer2convW,
				   b_init = layer2convb,
				   sparse_count = 3)
newlen = layer2conv.outputlen

layer3conv = ConvLayer(rng,
				   input = layer2conv.output,
				   filter_shape = (n_fmaps[3],n_fmaps[2],fmap_sizes[3],fmap_sizes[3],fmap_sizes[3]),
				   image_shape = (1,n_fmaps[2],newlen[0],newlen[1],newlen[2]),
				   W_init = layer3convW,
				   b_init = layer3convb,
				   sparse_count = 0,
				   softmax = 1)
newlen = layer2conv.outputlen

cost = layer3conv.negative_log_likelihood(y)

test_model = theano.function(inputs = [idx,idy,idz],
							outputs = [layer3conv.errors(y),layer3conv.p_y_given_x,layer3conv.y_pred], 
							givens = {x: test_data[0:num_channels,idx:idx+plen,idy:idy+plen,idz:idz+plen],
							z: test_truth[idx+offset/2:idx+plen-offset/2 +1, idy+offset/2:idy+plen-offset/2 +1,idz+offset/2:idz+plen-offset/2 +1]})

params = layer3conv.params + layer2conv.params + layer1conv.params + layer0conv.params
masks = layer3conv.masks + layer2conv.masks + layer1conv.masks + layer0conv.masks
grads = T.grad(cost,params)
#update only sparse elements
updates = [(param_i,param_i-learning_rate*grad_i*mask_i) for param_i,grad_i,mask_i in zip(params,grads,masks)]
train_model = theano.function([time_idx,idx,idy,idz],cost,updates = updates, 
							   givens = {x: train_data[time_idx,0:num_channels,idx:idx+plen,idy:idy+plen,idz:idz+plen],
										 z: train_truth[time_idx,idx+offset/2:idx+plen-offset/2 +1, idy+offset/2:idy+plen-offset/2 +1,idz+offset/2:idz+plen-offset/2 +1]})

############################################

maxXvalue = img_shape[0]-plen
maxYvalue = img_shape[1]-plen
maxZvalue = img_shape[2]-plen

xvalues = range(0,maxXvalue,numPred)
yvalues = range(0,maxYvalue,numPred)
zvalues = range(0,maxZvalue,numPred)

if((maxXvalue-1)%numPred != 0):
	xvalues = numpy.append(xvalues,maxXvalue-1)
if((maxYvalue-1)%numPred != 0):
	yvalues = numpy.append(yvalues,maxYvalue-1)
if((maxZvalue-1)%numPred != 0):
	zvalues = numpy.append(zvalues,maxZvalue-1)

############################################
localtime = time.asctime( time.localtime(time.time()) )
print "Start time is :", localtime
start_time  = time.clock()

myfile = open("logcost.txt", "a")           #File Name should be changed for every new run
    						
logcost = []
while(epoch < n_epochs) and (not done_looping):
	if(not resumeTraining):
		pat_idx = 0
	while(pat_idx < num_patients):
		if(not resumeTraining):
			tim_idx = 0
		while(tim_idx < TStamps[pat_idx]):
			end_time  = time.clock()
			print ('Training: epoch: %i, patient: %i, time stamp: %i %f\n' %(epoch+1,pat_idx+1,tim_idx+1,(end_time-start_time)/60))
			if(not resumeTraining):
				xi = 0
			while(xi < len(xvalues)):
				if(not resumeTraining):
					yi = 0
				while(yi < len(yvalues)):
					if(not resumeTraining):
						zi = 0
						resumeTraining = False
					while(zi < len(zvalues)):
						costTrain = train_model(tim_idx,xvalues[xi],yvalues[yi],zvalues[zi])
						logcost.append(costTrain)

						itr = itr + 1
						if itr % 100 == 0 :
							print 'Iteration: %i ' % (itr) 

						if(itr%saving_frequency == 0):
							print 'Saving model...'
							save_file = file('CNNmodel.pkl', 'wb')
							genVariables = [epoch,pat_idx,tim_idx,xi,yi,zi+numPred,itr+1]
							cPickle.dump(genVariables,save_file,protocol = cPickle.HIGHEST_PROTOCOL) 
							for i in xrange(len(params)):
								cPickle.dump(params[i].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)    
							save_file.close()
							for l in logcost:
								myfile.write("%f\n"%l)
							logcost = []

							print('Predicting for test patient')
							pstart_time = time.clock()
							Prediction = numpy.zeros(img_shape)

							for ix in xrange(len(xvalues)):
								for iy in xrange(len(yvalues)):
										for iz in xrange(len(zvalues)):
											errors,p_y_given_x,pred = test_model(xvalues[ix],yvalues[iy],zvalues[iz])
											pred = pred.reshape([numPred,numPred,numPred])
											Prediction[xvalues[ix]+offset/2:xvalues[ix]+plen-offset/2 +1,
													   yvalues[iy]+offset/2:yvalues[iy]+plen-offset/2 +1,
													   zvalues[iz]+offset/2:zvalues[iz]+plen-offset/2 +1] = pred
											p_y_given_x = p_y_given_x.transpose(1,0).reshape([num_classes,numPred,numPred,numPred])
											PostProbs[:,xvalues[ix]+offset/2:xvalues[ix]+plen-offset/2 +1,
													   	yvalues[iy]+offset/2:yvalues[iy]+plen-offset/2 +1,
													   	zvalues[iz]+offset/2:zvalues[iz]+plen-offset/2 +1] = p_y_given_x

							pend_time = time.clock()
							print >> sys.stderr, ('Prediction done and it took '+' %.2fm ' % ((pend_time - pstart_time) / 60.))
							#output nii file
							affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
							img = nib.Nifti1Image(Prediction, affine)
							img.set_data_dtype(numpy.int32)
							nib.save(img,'prediction.nii')
							for ix in numpy.arange(num_classes):
								img = nib.Nifti1Image(PostProbs[ix],affine)
								img.set_data_dtype(numpy.float32)
								nib.save(img,'postprobs' + str(ix+1) + '.nii')					

						zi = zi + 1
					yi = yi + 1
				xi = xi + 1			
			tim_idx = tim_idx + 1	
		pat_idx = pat_idx + 1					
		if(pat_idx < num_patients):
			data = numpy.zeros([numpy.max(TStamps),num_channels, img_shape[0],img_shape[1],img_shape[2]],dtype = theano.config.floatX)
			truth = numpy.zeros([numpy.max(TStamps),img_shape[0], img_shape[1],img_shape[2]],dtype = 'int32')
			for index in xrange(TStamps[pat_idx]):
				pat = ReadPat.new(pat_idx+1,index+1,num_channels)
				data[index,:,:,:,:] = numpy.asarray(pat.data,dtype = theano.config.floatX)
				truth[index,:,:,:] = numpy.asarray(pat.truth,dtype = 'int32')				
			train_data.set_value(data)
			train_truth.set_value(truth)
		
	epoch = epoch + 1			

end_time = time.clock()
print('Optimization complete.')
localtime = time.asctime( time.localtime(time.time()) )
print "End time is :", localtime
print >> sys.stderr, ('Training took '+' %.2fm ' % ((end_time - start_time) / 60.))

# Save model to file after whole optimization is done
save_file = file('FinalModel.pkl', 'wb')
for i in xrange(len(params)):
	cPickle.dump(params[i].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)    
save_file.close()

print('Predicting for test patient')
start_time = time.clock()
Prediction = numpy.zeros(img_shape)

for ix in xrange(len(xvalues)):
	for iy in xrange(len(yvalues)):
			for iz in xrange(len(zvalues)):
				errors,pred = test_model(xvalues[ix],yvalues[iy],zvalues[iz])
				pred = pred.reshape([numPred,numPred,numPred])
				Prediction[xvalues[ix]+offset/2:xvalues[ix]+plen-offset/2 +1,
						   yvalues[iy]+offset/2:yvalues[iy]+plen-offset/2 +1,
						   zvalues[iz]+offset/2:zvalues[iz]+plen-offset/2 +1] = pred

end_time = time.clock()
print >> sys.stderr, ('Prediction done and it took '+' %.2fm ' % ((end_time - start_time) / 60.))
#output nii file
affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
img = nib.Nifti1Image(Prediction, affine)
img.set_data_dtype(numpy.int32)
nib.save(img,'finalprediction.nii')
