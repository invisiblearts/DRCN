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