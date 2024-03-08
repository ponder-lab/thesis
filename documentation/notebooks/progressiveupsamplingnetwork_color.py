# %%
# !git clone -b dev https://github.com/nyikovicsmate/thesis

# %%
# %cd thesis

# %%
# !pip3 install -q gdown
# bsd500_gray.zip
# !gdown https://drive.google.com/uc?id=1O2tduoLX1DdP3VoLkAQfuv5ssFxf8LPc
# !unzip -q bsd500_gray.zip
# # bsd500_color.zip
# !gdown https://drive.google.com/uc?id=1buG1ziqMjy18gnpkuQjqk81XrLsGjT7p
# !unzip -q bsd500_color.zip
# # set14_color.zip
# !gdown https://drive.google.com/uc?id=1OiDs7jRm3NZCY6ghjyE12G5hVS0fG4EM
# !unzip -q set14_color.zip

# %%
# %tensorflow_version 2.x
from src.networks.supervised.progressive_upsampling_network import ProgressiveUpsamplingNetwork
from src.callbacks import *
from src.dataset import *
import numpy as np
import tensorflow as tf

# %%
batch_size = 20
seed = 1111
epochs = 2000
learning_rate = 1e-4

normalize = lambda x: np.asarray(x / 255.0, dtype=np.float32)
downsample = lambda x: np.array([cv2.resize(x_i, (x.shape[2]//2, x.shape[1]//2), interpolation=cv2.INTER_CUBIC) for x_i in x])

ds = HDFDataset("bsd500_140_140_color.h5").batch(batch_size).shuffle(seed).transform()
ds_hr_1 = ds.map(normalize)
ds_hr_0 = ds.map(downsample).map(normalize)
ds_lr = ds.map(downsample).map(downsample).map(normalize)

input_shape = (None, None, 3)
loss_func = tf.keras.losses.mse
cb = [TrainingCheckpointCallback(appendix="_color", save_freq=10),
       ExponentialDecayCallback(learning_rate, epochs/4, decay_rate=0.5)]

# %%
network = ProgressiveUpsamplingNetwork(input_shape=input_shape)
network.train(ds_lr, [ds_hr_0, ds_hr_1], None, epochs, learning_rate, cb)

# %%
from google.colab.patches import cv2_imshow

ds_ev_lr = DirectoryDataset("set14_70_70_color")\
    .map(downsample)\
    .map(normalize)
ds_ev_hr = DirectoryDataset("set14_70_70_color")\
    .map(normalize)

# load best network state
network.load_state("_color")

with ds_ev_lr as x, ds_ev_hr as y:
    x_batch = next(iter(x))
    y_batch = next(iter(y))
    # predict
    y_pred = network.predict(x_batch)
    print("   HR   ALIASED   NETWORK   ")
    for image_idx in range(len(x_batch)):
        aliased = tf.image.resize(x_batch[image_idx], size=tuple(y_batch[image_idx].shape[:2]), method="bicubic", antialias=True)
        img_0 = np.concatenate((y_batch[image_idx], aliased, y_pred[image_idx]), axis=1)
        cv2_imshow(img_0*255)
    # evaluate
    results = network.evaluate(y_batch, y_pred)

# %%
# PREDICT WITH THE TRAINING DATASET - TRANSFORMED

ds = HDFDataset("bsd500_70_70_color.h5").batch(10).transform()
ds_hr = ds.map(normalize)
ds_lr = ds.map(downsample).map(normalize)

with ds_lr.batch(10) as x, ds_hr.batch(10) as y:
    x_batch = next(iter(x))
    y_batch = next(iter(y))
    # predict
    y_pred = network.predict(x_batch)
    print("   HR   ALIASED   NETWORK   ")
    for image_idx in range(len(x_batch)):
        aliased = tf.image.resize(x_batch[image_idx], size=tuple(y_batch[image_idx].shape[:2]), method="bicubic", antialias=True)
        img_0 = np.concatenate((y_batch[image_idx], aliased, y_pred[image_idx]), axis=1)
        cv2_imshow(img_0*255)
    # evaluate
    results = network.evaluate(y_batch, y_pred)

# %%
# PREDICT WITH THE TRAINING DATASET - NON-TRANSFORMED

ds = HDFDataset("bsd500_70_70_color.h5").batch(10)
ds_hr = ds.map(normalize)
ds_lr = ds.map(downsample).map(normalize)

with ds_lr.batch(10) as x, ds_hr.batch(10) as y:
    x_batch = next(iter(x))
    y_batch = next(iter(y))
    # predict
    y_pred = network.predict(x_batch)
    print("   HR   ALIASED   NETWORK   ")
    for image_idx in range(len(x_batch)):
        aliased = tf.image.resize(x_batch[image_idx], size=tuple(y_batch[image_idx].shape[:2]), method="bicubic", antialias=True)
        img_0 = np.concatenate((y_batch[image_idx], aliased, y_pred[image_idx]), axis=1)
        cv2_imshow(img_0*255)
    # evaluate
    results = network.evaluate(y_batch, y_pred)

# %%
# !zip -r ./progressiveupsamplingnetwork_color.zip ./checkpoints/progressiveupsamplingnetwork_color

# %%
from google.colab.files import download

download("progressiveupsamplingnetwork_color.zip")

# %%
