import vapoursynth as vs
import h5py
import mvsfunc as mvf
import numpy as np
import gc
import random

def int_division(a, b):
    assert isinstance(a, int)
    assert isinstance(b, int)
    return int(a / b), a - b * (int(a / b))

def get_data_from_frame(frame, dim, num):
    assert isinstance(frame, vs.VideoFrame)
    assert isinstance(dim, int)
    arr = frame.get_read_array(0)
    returnList = []
    w = frame.width
    h = frame.height
    col = int(w / dim)
    row = int(h / dim)
    all = col * row
    index_list = random.sample(range(all), num)
    for i in index_list:
        r_i, c_i = int_division(i, col)
        out = np.array(arr, copy=False)[r_i * dim : (r_i + 1 ) * dim, c_i * dim : (c_i + 1) * dim]
        returnList.append(out)
    return returnList

prefix = 'train_DRCN'
suffix = '.h5'
labelOutput = prefix + '_label' + suffix
dataOutput = prefix + '_data' + suffix

data = '00003.m2ts'
dataDim = 41

# Get source and extract Y
core = vs.get_core()
labelClip = mvf.Depth(core.lsmas.LWLibavSource(data).std.ShufflePlanes(0, vs.GRAY), 32)
w = labelClip.width
h = labelClip.height
dataClip = labelClip.resize.Bicubic(int(w / 2), int(h / 2)).resize.Bicubic(w, h)
frameNum = dataClip.num_frames

sampleFrameNum = 3000
samplePerFrame = 50
sampleNum = sampleFrameNum * samplePerFrame
assert labelClip.num_frames >= sampleFrameNum

# Prepare HDF5 database
labelFile = h5py.File(labelOutput, 'w')
dataFile = h5py.File(dataOutput, 'w')
dataFile.create_dataset('data', (sampleNum, 1, dataDim, dataDim), 'single')
labelFile.create_dataset('label', (sampleNum, 1, dataDim, dataDim), 'single')

startloc = 0

# Get data from clip and write it to HDF5
numFrames = labelClip.num_frames
i = 0
currentSample = 0
rlist = random.sample(range(numFrames), sampleFrameNum)
while i < sampleFrameNum:
    print(str(i))
    currentFrame = rlist[i]
    currentDataFrame = dataClip.get_frame(currentFrame)
    currentLabelFrame = labelClip.get_frame(currentFrame)
    dataSubList = get_data_from_frame(currentDataFrame, dataDim, samplePerFrame)
    labelSubList = get_data_from_frame(currentLabelFrame, dataDim, samplePerFrame)
    m = 0
    while m < samplePerFrame:
        current_num = i * samplePerFrame + m
        dataFile['data'][current_num] = dataSubList[m]
        labelFile['label'][current_num] = labelSubList[m]
        m += 1
    i += 1
    del currentDataFrame, currentLabelFrame, dataSubList, labelSubList
    gc.collect()
