# The MIT License (MIT)
#
# Copyright (c) 2016 invisiblearts
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from keras.engine import Layer, InputSpec
import keras.backend as K
import numpy as np


class DRCN_Merge(Layer):
    def __init__(self, nbChannnels, **kwargs):
        self.supports_masking = False
        self.ch = nbChannnels
        self.input_spec = [InputSpec(shape=(None, self.ch, None, None))]
        super(DRCN_Merge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = K.variable(np.random.random(self.ch))
        assert len(input_shape) == 4
        tmp = list(input_shape)
        tmp[1] = 1
        self.batch_out_shape = tuple(tmp)
        self.trainable_weights = [self.alpha]

    def call(self, x, mask=None):
        tmp = self.alpha[0] * x[:, 0, :, :]
        for i in range(1, self.ch):
            tmp += self.alpha[i] * x[:, i, :, :]
        return K.reshape(tmp, self.batch_out_shape)

    def get_output_shape_for(self, input_shape):
        in_dim = list(input_shape)
        in_dim[1] = 1
        return tuple(in_dim)