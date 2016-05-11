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