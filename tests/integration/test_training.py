import os
import pytest
import tensorflow as tf
from pathlib import Path
from deepClassifier.entity import PrepareCallbacksConfig, TrainingConfig
from deepClassifier.components import PrepareCallback, Training
from deepClassifier.utils import get_size

        
class Test_Training_train:
    prepare_callbacks_config = PrepareCallbacksConfig (
        root_dir=Path('tests/data/prepare_callbacks/'),
        tensorboard_root_log_dir=Path('tests/data/prepare_callbacks/tensorboard_log_dir/'),
        checkpoint_model_filepath=Path('tests/data/prepare_callbacks/checkpoint_dir/model.h5')
    )
    training_config = TrainingConfig(
            root_dir=Path('tests/data/training/'),
            trained_model_path=Path('tests/data/training/model.h5'),
            updated_base_model_path=Path('tests/data/prepare_base_model/base_model_updated.h5'),
            training_data=Path("tests/data/data_ingestion/PetImages/"),
            params_epochs=1,
            params_batch_size=1,
            params_is_augmentation=True,
            params_image_size=[224, 224, 3]
        )

    def test_get_tb_ckpt_callbacks(self):
        prepare_callbacks = PrepareCallback(config=self.prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
        assert isinstance(callback_list, list)
        assert len(callback_list) == 2

    def test_get_base_model(self):
        training = Training(config=self.training_config)
        training.get_base_model()
        assert os.path.exists(self.training_config.updated_base_model_path)

    def test_train(self):
        prepare_callbacks = PrepareCallback(config=self.prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training = Training(config=self.training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)
        assert os.path.exists(self.prepare_callbacks_config.tensorboard_root_log_dir)
        # assert os.path.exists(self.prepare_callbacks_config.checkpoint_model_filepath)
        assert os.path.exists(self.training_config.training_data)
        assert os.path.exists(self.training_config.trained_model_path)
        # assert os.path.isfile(self.prepare_callbacks_config.checkpoint_model_filepath)
        assert len(os.listdir(self.prepare_callbacks_config.tensorboard_root_log_dir)) > 0
        assert len(os.listdir(self.training_config.training_data)) > 0
        assert get_size(self.training_config.trained_model_path) != 0
        # assert get_size(self.prepare_callbacks_config.checkpoint_model_filepath) != 0
