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
import h5py
import numpy as np
import json

FILENAME = 'DRCN_weights.hdf5'


def dataset_to_list(set):
    return np.array(set).tolist()


out = {
    'conv1_w': None,
    'conv1_b': None,
    'conv2_w': None,
    'conv2_b': None,
    'conv3_w': None,
    'conv3_b': None,
    'conv4_w': None,
    'conv4_b': None,
    'conv5_w': None,
    'conv5_b': None,
    'merge_w': None
}

a = h5py.File(FILENAME)
out['conv1_w'] = dataset_to_list(a['Embedding Net']['convolution2d_1_W'])
out['conv1_b'] = dataset_to_list(a['Embedding Net']['convolution2d_1_b'])
out['conv2_w'] = dataset_to_list(a['Embedding Net']['convolution2d_2_W'])
out['conv2_b'] = dataset_to_list(a['Embedding Net']['convolution2d_2_b'])
out['conv3_w'] = dataset_to_list(a['Inference Net']['convolution2d_3_W'])
out['conv3_b'] = dataset_to_list(a['Inference Net']['convolution2d_3_b'])
out['conv4_w'] = dataset_to_list(a['Reconstruction Net']['convolution2d_4_W'])
out['conv4_b'] = dataset_to_list(a['Reconstruction Net']['convolution2d_4_b'])
out['conv5_w'] = dataset_to_list(a['Reconstruction Net']['convolution2d_5_W'])
out['conv5_b'] = dataset_to_list(a['Reconstruction Net']['convolution2d_5_b'])
out['merge_w'] = dataset_to_list(a['drcn_merge_1']['param_0'])

with open('DRCN_weights.json', 'w') as fp:
    fp.write(json.dumps(out))