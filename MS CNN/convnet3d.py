import theano
import numpy
import theano.tensor as T

from conv3d2d import conv3d
from maxpool3d import max_pool_3d
epsilon = numpy.finfo(numpy.float32).eps

class ConvLayer(object):
        def __init__(self, rng, input, filter_shape, image_shape,W_init,b_init,sparse_count,softmax = 0):
                assert image_shape[1] == filter_shape[1]
                self.input = input
                fan_in = numpy.prod(filter_shape[1:])
                fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
                W_bound = numpy.sqrt(6. / (fan_in + fan_out))

                oneZeros = numpy.concatenate(([1],numpy.zeros(sparse_count)))
                x = numpy.insert(numpy.tile(oneZeros,filter_shape[2]-1),
                                 (filter_shape[2]-1)*(len(oneZeros)),1)
                y = numpy.insert(numpy.tile(oneZeros,filter_shape[3]-1),
                                 (filter_shape[3]-1)*(len(oneZeros)),1)
                z = numpy.insert(numpy.tile(oneZeros,filter_shape[4]-1),
                                 (filter_shape[4]-1)*(len(oneZeros)),1)
                mask = numpy.outer(numpy.outer(x,y),z).reshape(len(x),len(y),len(z))
                filter_shape = (filter_shape[0],
                                filter_shape[1],
                                (1 + sparse_count)*filter_shape[2] - sparse_count,
                                (1 + sparse_count)*filter_shape[3] - sparse_count,
                                (1 + sparse_count)*filter_shape[4] - sparse_count )
                self.Wmask = (numpy.ones(filter_shape)*mask).astype(theano.config.floatX)
                
                if W_init != None :
                    W_values = W_init
                else:
                    W_values = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size= filter_shape)*self.Wmask,
                                          dtype=theano.config.floatX)
                self.W = theano.shared(value = W_values, borrow=True)    

                if b_init != None :
                    b_values = b_init
                else:
                    b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
                self.b = theano.shared(value=b_values, borrow=True)
                self.bmask = numpy.ones((filter_shape[0],),dtype = theano.config.floatX)

                conv_out = conv3d(
                    signals = input.dimshuffle([0,2,1,3,4]),
                    filters=self.W.dimshuffle([0,2,1,3,4]),
                    signals_shape= [image_shape[i] for i in [0,2,1,3,4]],
                    filters_shape=[filter_shape[i] for i in [0,2,1,3,4]],
                    border_mode = 'valid'          
                ).dimshuffle([0,2,1,3,4])
                conv_out += self.b.dimshuffle('x',0,'x','x','x')

                self.outputlen = (image_shape[2]-filter_shape[2] +1,
                                  image_shape[3]-filter_shape[3] +1,
                                  image_shape[4]-filter_shape[4] +1)
                self.output = T.nnet.softplus(conv_out)
                self.params = [self.W, self.b]
                self.masks = [self.Wmask, self.bmask]
                self.num_points = T.prod(self.outputlen)

# initial shape = 1,3,img_shape
                if (softmax):
                    out = conv_out.reshape([conv_out.shape[1],self.num_points]).dimshuffle(1,0)
                    self.p_y_given_x = T.nnet.softmax(out)
                    self.y_pred = T.argmax(self.p_y_given_x,axis = 1)

        def negative_log_likelihood(self,y):
                return -T.mean(T.log(epsilon + self.p_y_given_x)[T.arange(self.num_points),y])

        def errors(self,y):
            return T.mean(T.neq(self.y_pred,y))



class PoolLayer(object):
        def __init__(self,input,image_shape, pool_size,sparse_count):

                #not implementing max pooling as of now. have to do with average pooling    
                oneZeros = numpy.concatenate(([1],numpy.zeros(sparse_count)))
                x = numpy.insert(numpy.tile(oneZeros,pool_size[0]-1),
                                 (pool_size[0]-1)*(len(oneZeros)),1)
                y = numpy.insert(numpy.tile(oneZeros,pool_size[1]-1),
                                 (pool_size[1]-1)*(len(oneZeros)),1)
                z = numpy.insert(numpy.tile(oneZeros,pool_size[2]-1),
                                 (pool_size[2]-1)*(len(oneZeros)),1)
                mask = numpy.outer(numpy.outer(x,y),z).reshape(len(x),len(y),len(z))
                mask = numpy.ones((1,1,len(x),len(y),len(z)))*mask
                self.pool_mask = mask.astype(theano.config.floatX)/numpy.prod(pool_size)

                frame_shape = input.shape[-3:]
                batch_size = T.shape_padright(T.prod(input.shape[:-3]),1)
                new_shape = T.cast(T.join(0, batch_size,
                                        T.as_tensor([1,]), 
                                        frame_shape), 'int32')
                filter_shape = (1,1,len(x),len(y),len(z))
                input_5d = T.reshape(input,new_shape,ndim = 5)         
                image_shape = (image_shape[0]*image_shape[1],
                               1,
                               image_shape[2],
                               image_shape[3],
                               image_shape[4])
                avg_out = conv3d(
                    signals = input_5d.dimshuffle([0,2,1,3,4]),
                    filters = self.pool_mask.transpose(0,2,1,3,4),
                    signals_shape = [image_shape[i] for i in [0,2,1,3,4]],
                    filters_shape = [filter_shape[i] for i in [0,2,1,3,4]],
                    border_mode = 'valid').dimshuffle([0,2,1,3,4])
                outshp = T.join(0,input.shape[:-3],avg_out.shape[-3:])
                avg_out = T.reshape(avg_out,outshp,ndim = 5)

                self.outputlen = (image_shape[2] - len(x) + 1,
                                  image_shape[3] - len(y) + 1,
                                  image_shape[4] - len(z) + 1)
                self.output = avg_out

def gradient_updates_momentum(cost, params, grads, masks, learning_rate, momentum):

    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param,grad,mask in zip(params,grads,masks):
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)*mask))
    return updates
