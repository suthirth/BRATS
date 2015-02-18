import theano
import numpy
import theano.tensor as T

from conv3d2d import conv3d
from maxpool3d import max_pool_3d

class ConvPoolLayer(object):
        def __init__(self, rng, input, filter_shape, image_shape,pool_size = (2,2,2)):
                assert image_shape[1] == filter_shape[1]
                self.input = input
                fan_in = numpy.prod(filter_shape[1:])
                fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                           numpy.prod(pool_size))
                W_bound = numpy.sqrt(6. / (fan_in + fan_out))
                self.W = theano.shared(
                    numpy.asarray(
                        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                        dtype=theano.config.floatX
                    ),
                    borrow=True
                )
                b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
                self.b = theano.shared(value=b_values, borrow=True)
                conv_out = conv3d(
                    signals = input.dimshuffle([0,2,1,3,4]),
                    filters=self.W,
                    signals_shape= [image_shape[i] for i in [0,2,1,3,4]],
                    filters_shape=[filter_shape[i] for i in [0,2,1,3,4]],
                    border_mode = 'valid'          
                ).dimshuffle([0,2,1,3,4])

                pooled_out = max_pool_3d(input = conv_out,ds = pool_size, ignore_border = True)

                self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
                self.params = [self.W, self.b]