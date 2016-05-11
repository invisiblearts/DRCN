# The MIT License (MIT)
#
# Copyright (c) 2016 invisiblearts
# Copyright (c) 2016 mawen1250
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

import vapoursynth as vs
import h5py
import mvsfunc as mvf
import numpy as np
import math
import gc
import random


def resample(clip, scale=2, linear_scale=False, down=5, upfilter='bicubic'):
    assert isinstance(clip, vs.VideoNode)

    core = vs.get_core()
    sw = clip.width
    sh = clip.height
    dw = math.floor(sw / scale + 0.5)
    dh = math.floor(sh / scale + 0.5)

    # gamma to linear
    if linear_scale:
        clip = clip.resize.Bicubic(transfer_s='linear', transfer_in_s='709')

    # down-sampling
    if down == 0:
        clip = clip.resize.Point(dw, dh)
    elif down == 1:
        clip = clip.resize.Bilinear(dw, dh)
    elif down == 2:
        clip = clip.resize.Spline16(dw, dh)
    elif down == 3:
        clip = clip.resize.Spline36(dw, dh)
    elif down == 4:
        clip = clip.resize.Lanczos(dw, dh, filter_param_a=3)
    elif down == 5:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=-0.5, filter_param_b=0.25)
    elif down == 6:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=0, filter_param_b=0.5) # Catmull-Rom
    elif down == 7:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=1/3, filter_param_b=1/3) # Mitchell-Netravali
    elif down == 8:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=0.3782, filter_param_b=0.3109) # Robidoux
    elif down == 9:
        clip = clip.resize.Bicubic(dw, dh, filter_param_a=1, filter_param_b=0) # SoftCubic100
    else:
        raise ValueError('unknown \'down\'')

    # up-sampling
    if upfilter == 'bicubic':
        clip = clip.resize.Bicubic(sw, sh, filter_param_a=0, filter_param_b=0.5)
    elif upfilter == 'point':
        clip = clip.resize.Point(sw, sh)
    else:
        raise ValueError('unknown \'upfilter\'')

    # linear to gamma
    if linear_scale:
        clip = clip.resize.Bicubic(transfer_s='709', transfer_in_s='linear')

    return clip


def int_division(a, b):
    assert isinstance(a, int)
    assert isinstance(b, int)
    return a // b, a % b


def get_data_from_frame(frame, num, planes, dim):
    assert isinstance(frame, vs.VideoFrame)
    assert isinstance(num, int)
    assert isinstance(planes, int)
    assert isinstance(dim, int)
    arr = np.array([frame.get_read_array(p) for p in range(planes)], copy=False)
    return_list = []
    w = frame.width
    h = frame.height
    col = w // dim
    row = h // dim
    all = col * row
    index_list = random.sample(range(all), num)
    for i in index_list:
        r_i, c_i = int_division(i, col)
        out = arr[:, r_i * dim : (r_i + 1) * dim, c_i * dim : (c_i + 1) * dim]
        return_list.append(out)
    return return_list


def shuffle_together(lists):
    assert isinstance(lists, list)
    state = random.getstate()

    for l in lists:
        assert isinstance(l, list)
        random.setstate(state)
        random.shuffle(l)

prefix = 'train_DRCN'
suffix = '.h5'
data_output = prefix + '_data' + suffix
label_output = prefix + '_label' + suffix

useRGB = False
scale = 2
linear_scale = False
upfilter = 'bicubic'

# data = '00003.m2ts'
data = r'I:\Anime\The Garden of Words\BDROM\BDMV\STREAM\00000.m2ts'
data_dim = 41
planes = 3 if useRGB else 1

# Get source and do format conversion
core = vs.get_core()
label_clip = core.lsmas.LWLibavSource(data)
if useRGB:
    label_clip = mvf.ToRGB(label_clip, depth=32)
else:
    label_clip = mvf.Depth(label_clip.std.ShufflePlanes(0, vs.GRAY), 32)

# Prepare data
down_lists = [d for d in range(1, 8)]
data_clip = core.std.Interleave([resample(label_clip, scale, linear_scale, d, upfilter) for d in down_lists])
label_clip = core.std.Interleave([label_clip for d in down_lists])

w = data_clip.width
h = data_clip.height
nb_frame = data_clip.num_frames
assert w == label_clip.width
assert h == label_clip.height
assert nb_frame == label_clip.num_frames

nb_sample_frame = 10000
nb_sample_per_frame = 32
nb_sample = nb_sample_frame * nb_sample_per_frame
assert nb_frame >= nb_sample_frame

# Prepare HDF5 database
data_file = h5py.File(data_output, 'w')
label_file = h5py.File(label_output, 'w')
data_file.create_dataset('data', (nb_sample, planes, data_dim, data_dim), 'single', fillvalue=0)
label_file.create_dataset('label', (nb_sample, planes, data_dim, data_dim), 'single', fillvalue=0)
data_set = data_file['data']
label_set = label_file['label']

# Get data from clip and write to HDF5
frame_list = random.sample(range(nb_frame), nb_sample_frame)
frame_list.sort()
index_list = [i for i in range(nb_sample)]
random.shuffle(index_list)
i = 0
for f in range(nb_sample_frame):
    nb_frame_current = frame_list[f]
    print('{:>6}: extracting from frame {:>6}'.format(f, nb_frame_current))
    data_frame_current = data_clip.get_frame(nb_frame_current)
    label_frame_current = label_clip.get_frame(nb_frame_current)
    sub_data_list = get_data_from_frame(data_frame_current, nb_sample_per_frame, planes, data_dim)
    sub_label_list = get_data_from_frame(label_frame_current, nb_sample_per_frame, planes, data_dim)
    for s in range(nb_sample_per_frame):
        data_set[index_list[i]] = sub_data_list[s]
        label_set[index_list[i]] = sub_label_list[s]
        i += 1
    del data_frame_current, label_frame_current, sub_data_list, sub_label_list
    gc.collect()
