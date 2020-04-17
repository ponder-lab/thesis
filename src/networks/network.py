import pickle
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple

import numpy as np
import tensorflow as tf

from src.config import *
from src.dataset import Dataset


class Network(ABC):

    def __init__(self,
                 model: tf.keras.models.Model):
        self.model = model
        self._state = self.NetworkState()

    class NetworkState:

        def __init__(self):
            self._train_time: float = 0.0
            self.epochs: int = 0
            self.train_loss: float = 0

        @property
        def train_time(self) -> str:
            hours, rem = divmod(self._train_time, 3600)
            minutes, rem = divmod(rem, 60)
            secounds = rem
            return f"{hours:02.0f}h:{minutes:02.0f}m:{secounds:02.0f}s"

        @train_time.setter
        def train_time(self, value: int):
            """
            :param value: The delta training time in seconds. The total training time will be incremented by this value.
            """
            self._train_time += value

        def __str__(self):
            s = f"Train time: {self.train_time}{chr(10)}" + \
                f"Epochs:     {self.epochs}{chr(10)}" + \
                f"Loss:       {self.train_loss:.4f}"
            return s

    @property
    def state(self):
        return self._state

    def save_state(self):
        # make sure the save directory exists
        checkpoint_dir_path = ROOT_PATH.joinpath("checkpoints")
        if not checkpoint_dir_path.exists() or not checkpoint_dir_path.is_dir():
            LOGGER.warning(f"Save directory {checkpoint_dir_path} does not exist. Creating it.")
            checkpoint_dir_path.mkdir(parents=False, exist_ok=False)
        # save the keras model to Tensorflow SavedModel format
        model_dir_path = checkpoint_dir_path.joinpath(str.lower(self.__class__.__name__))
        self.model.save(filepath=str(model_dir_path), overwrite=True, include_optimizer=True, save_format="tf")
        # also save the state of the model
        state_file_path = model_dir_path.joinpath("state.dat")
        with open(str(state_file_path), "wb") as f:
            pickle.dump(self.state, f)

    def load_state(self):
        # make sure the save directory exists
        checkpoint_dir_path = ROOT_PATH.joinpath("checkpoints")
        if not checkpoint_dir_path.exists() or not checkpoint_dir_path.is_dir():
            raise FileExistsError(f"Save directory {checkpoint_dir_path} does not exist.")
        # load the keras model from Tensorflow SavedModel format
        model_dir_path = checkpoint_dir_path.joinpath(str.lower(self.__class__.__name__))
        self.model = tf.keras.models.load_model(filepath=str(model_dir_path))
        # also load the previous state of the model
        state_file_path = model_dir_path.joinpath("state.dat")
        with open(str(state_file_path), "rb") as f:
            self._state = pickle.load(f)

    @abstractmethod
    def train(self, dataset_x: Dataset, dataset_y: Union[Dataset, List[Dataset]], loss_func: Callable[[np.ndarray, np.ndarray], float], epochs: int, learning_rate: float):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        :param x: The batch of LR input images.
        :param args, kwargs: Optional arguments for predictions. Possible values:
                        - int or float
                        A positive number defining the desired upsampling factor. (default: 2)
                        - Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]
                        A tuple defining the exact shape of the desired output. Accepted formats are
                        (height, width) or (height, width, depth) or (batch, height, width, depth).
                    Keywords for kwargs don't matter, input is determined solely by it's tpye. If none of
                    the above input types are specified, predictions are carried out using the default
                    upsampling factor of 2.
        :return: The predicted image batch with values [0-1].
        """
        pass

    @staticmethod
    def _parse_predict_optionals(x: np.ndarray, args, kwargs) -> Tuple[int, int]:
        """Helper function for parsing the optional arguments given to the `predict` function.

        :return: The determined upsampling shape as a (height, width) tuple.
        """
        default_upsampling_factor = 2
        assert 3 <= len(x.shape) <= 4, "`x` should be a (batch, height, width, depth) or (height, width, depth) shaped array."
        x_size = (x.shape[1], x.shape[2]) if len(x.shape) == 4 else (x.shape[0], x.shape[1])
        size = (int(x_size[0] * default_upsampling_factor), int(x_size[1] * default_upsampling_factor))   # the default size
        params = [] + list(args) + list(kwargs.values())
        if len(params) == 0:
            # use default upsampling factor
            LOGGER.info(f"Predicting using the default upsampling factor of {default_upsampling_factor}.")
        elif len(params) == 1:
            if type(params[0]) == int or type(params[0]) == float:
                LOGGER.info(f"Predicting using the supplied upsampling factor of {params[0]}.")
                size = (int(x_size[0] * params[0]), int(x_size[1] * params[0]))
            elif type(params[0]) == tuple:
                assert 2 <= len(params[0]) <= 4, f"Desired output size dim should be between 2 and 4, got {len(params[0])}"
                size = (params[0][1], params[0][2]) if len(params[0]) == 4 else (params[0][0], params[0][1])
                LOGGER.info(f"Predicting using the supplied size parameter {size}.")
            else:
                raise TypeError("The optional input parameter type did not match any of the acceptable types (int,float, tuple).")
        else:
            raise ValueError("Found more than 1 optional input parameters.")
        return size

    @staticmethod
    def evaluate(y: np.ndarray, y_pred: np.ndarray):
        """
        :param y: The batch of HR images.
        :param y_pred: The corresponding batch of predicted images.
        :return:
        """
        _len = len(y_pred)
        assert _len == len(y)
        ssim_list = []
        total_variation_list = []
        psnr_list = []
        mean_squared_error_list = []
        mean_absolute_error_list = []
        for i in range(_len):
            y_i_tensor = tf.convert_to_tensor(y[i], dtype=tf.float32)
            y_pred_i_tensor = tf.convert_to_tensor(y_pred[i], dtype=tf.float32)
            ssim_list.append(tf.image.ssim(y_i_tensor, y_pred_i_tensor, max_val=1).numpy())
            total_variation_list.append(tf.image.total_variation(y_pred_i_tensor).numpy())
            psnr_list.append(tf.image.psnr(y_i_tensor, y_pred_i_tensor, max_val=1).numpy())
            mean_squared_error_list.append(np.sum(tf.losses.mean_squared_error(y_i_tensor, y_pred_i_tensor).numpy()))
            mean_absolute_error_list.append(np.sum(tf.losses.mean_absolute_error(y_i_tensor, y_pred_i_tensor).numpy()))

        ssim_list = np.array(ssim_list)
        total_variation_list = np.array(total_variation_list)
        psnr_list = np.array(psnr_list)
        mean_squared_error_list = np.array(mean_squared_error_list)
        mean_absolute_error_list = np.array(mean_absolute_error_list)

        per_image_results = [
            {"ssim": ssim, "total_variation": tv, "psnr": psnr, "mean_squared_error": mse, "mean_absolute_error": mae}
            for ssim, tv, psnr, mse, mae
            in zip(ssim_list, total_variation_list, psnr_list, mean_squared_error_list, mean_absolute_error_list)]

        _max = {"ssim": np.amax(ssim_list),
               "total_variation": np.amax(total_variation_list),
               "psnr": np.amax(psnr_list),
               "mean_squared_error": np.amax(mean_squared_error_list),
               "mean_absolute_error": np.amax(mean_absolute_error_list)}

        _min = {"ssim": np.amin(ssim_list),
               "total_variation": np.amin(total_variation_list),
               "psnr": np.amin(psnr_list),
               "mean_squared_error": np.amin(mean_squared_error_list),
               "mean_absolute_error": np.amin(mean_absolute_error_list)}

        _avg = {"ssim": np.average(ssim_list),
               "total_variation": np.average(total_variation_list),
               "psnr": np.average(psnr_list),
               "mean_squared_error": np.average(mean_squared_error_list),
               "mean_absolute_error": np.average(mean_absolute_error_list)}

        print("SSIM")
        print(f"max: [{np.where(ssim_list == _max['ssim'])[0][0]}] {_max['ssim']:.2f}    "
              f"min: [{np.where(ssim_list == _min['ssim'])[0][0]}] {_min['ssim']:.2f}    "
              f"avg: {_avg['ssim']:.2f}")
        print("TOTAL_VARIATION")
        print(f"max: [{np.where(total_variation_list == _max['total_variation'])[0][0]}] {_max['total_variation']:.2f}    "
              f"min: [{np.where(total_variation_list == _min['total_variation'])[0][0]}] {_min['total_variation']:.2f}    "
              f"avg: {_avg['total_variation']:.2f}")
        print("PSNR")
        print(f"max: [{np.where(psnr_list == _max['psnr'])[0][0]}] {_max['psnr']:.2f}    "
              f"min: [{np.where(psnr_list == _min['psnr'])[0][0]}] {_min['psnr']:.2f}    "
              f"avg: {_avg['psnr']:.2f}")
        print("MEAN_SQUARED_ERROR")
        print(f"max: [{np.where(mean_squared_error_list == _max['mean_squared_error'])[0][0]}] {_max['mean_squared_error']:.2f}    "
              f"min: [{np.where(mean_squared_error_list == _min['mean_squared_error'])[0][0]}] {_min['mean_squared_error']:.2f}    "
              f"avg: {_avg['mean_squared_error']:.2f}")
        print("MEAN_ABSOLUTE_ERROR")
        print(f"max: [{np.where(mean_absolute_error_list == _max['mean_absolute_error'])[0][0]}] {_max['mean_absolute_error']:.2f}    "
              f"min: [{np.where(mean_absolute_error_list == _min['mean_absolute_error'])[0][0]}] {_min['mean_absolute_error']:.2f}    "
              f"avg: {_avg['mean_absolute_error']:.2f}")

        return per_image_results


