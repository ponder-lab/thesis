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
from src.callbacks import *
from src.dataset import *
import numpy as np
import tensorflow as tf
import cv2

from src.networks.reinforced.reinforced_network import ReinforcedNetwork

# %%
batch_size = 50
seed = 1111
epochs = 1800
learning_rate = 1e-4

normalize = lambda x: np.asarray(x / 255.0, dtype=np.float32)
downsample = lambda x: np.array([cv2.resize(x_i, (x.shape[2]//2, x.shape[1]//2), interpolation=cv2.INTER_CUBIC) for x_i in x])

ds = HDFDataset("bsd500_70_70_gray.h5").batch(batch_size).shuffle(seed)
ds_hr = ds.map(normalize)
ds_lr = ds.map(normalize)

loss_func = tf.keras.losses.mse
cb = [TrainingCheckpointCallback(appendix="_half", save_freq=10),
       ExponentialDecayCallback(learning_rate, epochs/6, decay_rate=0.5)]

# %%
def custom_noise_func(images: np.ndarray) -> np.ndarray:
    """
    Adds noise to image.
    :param images: A batch of images, 4D array (batch, height, width, channels)
    :return: The noisy batch of input images.
    """
    fill_value = 0.5  # SET THE FILL VALUE TO 0.5
    try:
        # this will fail unless there is exactly 4 dimensions to unpack from
        batch, height, width, channels = images.shape
    except ValueError:
        raise TypeError(f"Image must be a 4D numpy array. Got shape {images.shape}")
    if channels == 1:
        for img in images:
            for h in range(height):
                if h % 2 == 0:
                    img[h][0::2] = [fill_value]
                else:
                    img[h][1::2] = [fill_value]
    elif channels == 3:
        for img in images:
            for h in range(height):
                if h % 2 == 0:
                    img[h][0::2] = [fill_value, fill_value, fill_value]
                else:
                    img[h][1::2] = [fill_value, fill_value, fill_value]
    else:
        raise ValueError(f"Unsupported number of image dimensions, got {channels}")
    return images

# %%
network = ReinforcedNetwork()
# bind the custom noise function
network.noise_func = custom_noise_func
network.train(ds_lr, ds_hr, None, epochs, learning_rate, cb)

# %%
from google.colab.patches import cv2_imshow

ds_ev_lr = DirectoryDataset("set14_70_70_color").map(normalize)
ds_ev_hr = DirectoryDataset("set14_70_70_color").map(normalize)

# load best network state
network.load_state("_half")

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
# PREDICT WITH THE TRAINING DATASET - COLORED

ds = HDFDataset("bsd500_70_70_color.h5").batch(10)
ds_hr = ds.map(normalize)
ds_lr = ds.map(normalize)

with ds_lr as x, ds_hr as y:
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
# PREDICT WITH THE TRAINING DATASET - GRAY

ds = HDFDataset("bsd500_70_70_gray.h5").batch(10)
ds_hr = ds.map(normalize)
ds_lr = ds.map(normalize)

with ds_lr as x, ds_hr as y:
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
# !zip -r reinforcednetwork_half.zip ./checkpoints/reinforcednetwork_half

# %%
from google.colab.files import download

download("reinforcednetwork_half.zip")

# %%


# %%


# %%


# %%


# %%
