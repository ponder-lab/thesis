# %%
# !git clone -b dev https://github.com/nyikovicsmate/thesis

# %%
# %cd thesis

# %%
# !pip3 install -q gdown
# bsd500_gray.zip
# !gdown https://drive.google.com/uc?id=1O2tduoLX1DdP3VoLkAQfuv5ssFxf8LPc
# !unzip -q bsd500_gray.zip
# bsd500_color.zip
# !gdown https://drive.google.com/uc?id=1buG1ziqMjy18gnpkuQjqk81XrLsGjT7p
# !unzip -q bsd500_color.zip
# set14_color.zip
# !gdown https://drive.google.com/uc?id=1OiDs7jRm3NZCY6ghjyE12G5hVS0fG4EM
# !unzip -q set14_color.zip

# %%
# %tensorflow_version 2.x
from src.callbacks import *
from src.dataset import *
import numpy as np
import tensorflow as tf

from src.networks.adversarial.iterative_sampling_network import AdversarialIterativeSamplingNetwork

# %%
seed = 1111

normalize = lambda x: np.asarray(x / 255.0, dtype=np.float32)
downsample = lambda x: np.array([cv2.resize(x_i, (x.shape[2]//2, x.shape[1]//2), interpolation=cv2.INTER_CUBIC) for x_i in x])

ds = HDFDataset("bsd500_70_70_color.h5").shuffle(seed).transform()
ds_g_hr = ds.batch(50).map(normalize)
ds_g_lr = ds.batch(50).map(downsample).map(normalize)
ds_d_hr = ds.batch(50).map(normalize)
ds_d_lr = ds.batch(50).map(downsample).map(normalize)

# cb_g = [TrainingCheckpointCallback(appendix="_ad_iter", save_freq=10),
#        ExponentialDecayCallback(learning_rate, epochs, decay_rate=0.9)]
# cb_d = [TrainingCheckpointCallback(appendix="_ad_iter", save_freq=10),
#        ExponentialDecayCallback(learning_rate, epochs, decay_rate=0.9)]

# %%
network = AdversarialIterativeSamplingNetwork((None, None, 3))
network.train(ds_g_lr, ds_g_hr,
              ds_d_lr, ds_d_hr,
              generator_epochs=1800,
              discriminator_epochs=1200,
              alternating_ratio=20,
              generator_lr=1e-6,
              discriminator_lr=1e-6,
              generator_callbacks=[],
              discriminator_callbacks=[])

# %%
from google.colab.patches import cv2_imshow

ds_ev_lr = DirectoryDataset("set14_70_70_color").map(downsample).map(normalize)
ds_ev_hr = DirectoryDataset("set14_70_70_color").map(normalize)

network.save_state(generator_appendix="_ad_iter", discriminator_appendix="_ad_iter")
# load best network state
network.load_state(generator_appendix="_ad_iter", discriminator_appendix="_ad_iter")

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
# %cd checkpoints/
# !zip -r iterative_adversarial.zip ./discriminatornetwork_ad_iter ./iterativesamplingnetwork_ad_iter
# %cd ..

# %%
from google.colab.files import download

download("./checkpoints/iterative_adversarial.zip")

# %%


# %%


# %%
