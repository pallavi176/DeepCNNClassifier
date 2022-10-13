import os
import pytest
from pathlib import Path
from deepClassifier.entity import PrepareCallbacksConfig
from deepClassifier.components import PrepareCallback

class Test_PrepareCallbacks_ckpt:
    prepare_callbacks_config = PrepareCallbacksConfig (
        root_dir='tests/data/prepare_callbacks/',
        tensorboard_root_log_dir='tests/data/prepare_callbacks/tensorboard_log_dir/',
        checkpoint_model_filepath='tests/data/prepare_callbacks/checkpoint_dir/model.h5'
    )

    def test_get_tb_ckpt_callbacks(self):
        prepare_callbacks = PrepareCallback(config=self.prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
        assert isinstance(callback_list, list)
        assert len(callback_list) == 2
