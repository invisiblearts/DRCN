from keras.models import Sequential, Model
from keras.layers import Convolution2D, Input, merge, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils.io_utils import HDF5Matrix
from keras.optimizers import Adam
from drcn_merge import DRCN_Merge

BATCH_SIZE = 1


input_data = Input((1, 41, 41), name='data')

def func_iterator(x, func, times):
    assert isinstance(times, int)
    if times == 1:
        return func(x)
    return func_iterator(x, func, times-1)


def conv():
    return Convolution2D(256, 3, 3, 'he_normal', border_mode='same', activation='relu')

embed_net = Sequential([Activation('linear', input_shape=(1, 41, 41)), conv(), conv()], name='Embedding Net')
infer_net = Sequential([Activation('linear', input_shape=(256, 41, 41)), conv()], name='Inference Net')
recons_net = Sequential([Activation('linear', input_shape=(256, 41, 41)), conv(), Convolution2D(1, 3, 3, 'he_normal', border_mode='same')], name='Reconstruction Net')

features = embed_net(input_data)
recurrence_list = []
reconstruct_list = []
for i in range(10):
    recurrence_list.append(func_iterator(features, infer_net, i+1))
    reconstruct_list.append(merge([recons_net(recurrence_list[i]), input_data]))
merged = merge(reconstruct_list, mode='concat', concat_axis=1)
DRCNMerge = DRCN_Merge(10)
out = DRCNMerge(merged)

DRCN_Model = Model(input=input_data, output=out, name='DRCN Final Model')
DRCN_Model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999), loss='mae')

train_data = HDF5Matrix('train_DRCN_data.h5', 'data', 0, 147000)
train_label = HDF5Matrix('train_DRCN_label.h5', 'label', 0, 147000)
test_data = HDF5Matrix('train_DRCN_data.h5', 'data', 147000, 150000)
test_label = HDF5Matrix('train_DRCN_label.h5', 'label', 147000, 150000)

with open('DRCN.yaml', 'w') as fp:
    fp.write(DRCN_Model.to_yaml())


hist = DRCN_Model.fit(
    train_data, train_label,
    batch_size=BATCH_SIZE, nb_epoch=200,
    validation_data=[test_data, test_label], shuffle='batch',
    callbacks=[ModelCheckpoint('DRCN_weights.{epoch:02d}-{val_loss:.6f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=False, mode='auto')])

DRCN_Model.save_weights('DRCN_weights.h5')

with open('DRCN_history.txt', 'w') as fp:
    fp.write(str(hist.history))
