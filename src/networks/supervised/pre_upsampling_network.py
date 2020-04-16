import contextlib

import tensorflow as tf
import numpy as np

from src.config import *
from src.dataset import Dataset
from src.networks.network import Network
from src.models.supervised import PreUpsamplingModel


class PreUpsamplingNetwork(Network):

    def __init__(self):
        model = PreUpsamplingModel()
        super().__init__(model)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # TODO
        pass

    @tf.function
    def _train_step(self, x, y, optimizer, loss_func):
        with tf.GradientTape() as tape:
            _shape = tf.shape(y)  # expecting 4D tensor in channel_last format
            x = tf.image.resize(x, (_shape[1], _shape[2]), tf.image.ResizeMethod.BICUBIC)
            y_pred = self.model(x)
            loss = loss_func(y, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.reduce_sum(loss)

    def train(self, dataset_x, dataset_y, loss_func, epochs, learning_rate):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                       decay_steps=epochs,
                                                                       decay_rate=0.9,
                                                                       staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # threat a single value as a list regardless
        if isinstance(dataset_y, Dataset):
            dataset_y = [dataset_y]
        # open the datasets
        with contextlib.ExitStack() as stack:
            stack.enter_context(dataset_x)
            for dataset in dataset_y:
                stack.enter_context(dataset)
            iter_x = dataset_x.as_numpy_iterator()
            iter_y = [dataset.as_numpy_iterator() for dataset in dataset_y]
            # train
            e_idx = 0
            while e_idx < epochs:
                start_sec = time.time()
                train_loss = None
                random_y_idx = np.random.randint(len(dataset_y))
                # load the data
                try:
                    x = iter_x.next()
                    for idx in range(len(iter_y)):
                        if idx == random_y_idx:
                            y = iter_y[idx].next()
                        else:
                            iter_y[idx].advance()
                    # determine the scaling factor
                    # the dataset comprises of 4D vectors (batch_size, height, width, depth)
                    # scaling_factor = (y.shape[1] / x.shape[1], y.shape[2] / x.shape[2])
                    train_loss += self._train_step(x, y, optimizer, loss_func)
                    # update state
                    delta_sec = time.time() - start_sec
                    self.state.epochs += 1
                    self.state.train_loss = train_loss.numpy()
                    self.state.train_time = delta_sec
                    LOGGER.info(f"Epoch: {e_idx} train_loss: {train_loss:.2f}")
                    if e_idx > 0 and e_idx % 100 == 0:
                        LOGGER.info(f"Saving state after {e_idx} epochs.")
                        self.save_state()
                    e_idx += 1
                except StopIteration:
                    # reset iterators
                    iter_x.reset()
                    for _iter in iter_y:
                        _iter.reset()
