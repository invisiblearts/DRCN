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
    returnList = []
    w = frame.width
    h = frame.height
    col = w // dim
    row = h // dim
    all = col * row
    index_list = random.sample(range(all), num)
    for i in index_list:
        r_i, c_i = int_division(i, col)
        out = arr[:, r_i * dim : (r_i + 1) * dim, c_i * dim : (c_i + 1) * dim]
        returnList.append(out)
    return returnList

def shuffle_together(lists):
    assert isinstance(lists, list)
    state = random.getstate()

    for l in lists:
        assert isinstance(l, list)
        random.setstate(state)
        random.shuffle(l)

prefix = 'train_DRCN'
suffix = '.h5'
labelOutput = prefix + '_label' + suffix
dataOutput = prefix + '_data' + suffix

useRGB = False
scale = 2
linear_scale = False
upfilter = 'bicubic'

#data = '00003.m2ts'
data = r'I:\Anime\The Garden of Words\BDROM\BDMV\STREAM\00000.m2ts'
dataDim = 41
planes = 3 if useRGB else 1

# Get source and do format conversion
core = vs.get_core()
labelClip = core.lsmas.LWLibavSource(data)
if useRGB:
    labelClip = mvf.ToRGB(labelClip, depth=32)
else:
    labelClip = mvf.Depth(labelClip.std.ShufflePlanes(0, vs.GRAY), 32)

# Prepare data
down_lists = [d for d in range(1, 8)]
dataClip = core.std.Interleave([resample(labelClip, scale, linear_scale, d, upfilter) for d in down_lists])
labelClip = core.std.Interleave([labelClip for d in down_lists])

w = dataClip.width
h = dataClip.height
frameNum = dataClip.num_frames
assert w == labelClip.width
assert h == labelClip.height
assert frameNum == labelClip.num_frames

sampleFrameNum = 10000
samplePerFrame = 32
sampleNum = sampleFrameNum * samplePerFrame
assert frameNum >= sampleFrameNum

# Prepare HDF5 database
dataFile = h5py.File(dataOutput, 'w')
labelFile = h5py.File(labelOutput, 'w')
dataFile.create_dataset('data', (sampleNum, planes, dataDim, dataDim), 'single')
labelFile.create_dataset('label', (sampleNum, planes, dataDim, dataDim), 'single')

# Get data from clip
dataList = []
labelList = []
rlist = random.sample(range(frameNum), sampleFrameNum)
rlist.sort()
for i in range(sampleFrameNum):
    currentFrame = rlist[i]
    print('{:>6}: extracting from frame {:>6}'.format(i, currentFrame))
    currentDataFrame = dataClip.get_frame(currentFrame)
    currentLabelFrame = labelClip.get_frame(currentFrame)
    dataList += get_data_from_frame(currentDataFrame, samplePerFrame, planes, dataDim)
    labelList += get_data_from_frame(currentLabelFrame, samplePerFrame, planes, dataDim)
    del currentDataFrame, currentLabelFrame
    gc.collect()
shuffle_together([dataList, labelList])
dataList = np.concatenate(tuple(dataList))
labelList = np.concatenate(tuple(labelList))

# Write data to HDF5
dataFile['data'].write_direct(dataList)
labelFile['label'].write_direct(labelList)
del dataList
del dataList
