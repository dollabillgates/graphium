# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_architecture, load_predictor, load_trainer

from goli.trainer.predictor import PredictorModule


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
os.chdir(MAIN_DIR)

MODEL_FILE = "models_checkpoints/micro_ZINC/model.ckpt"
CONFIG_FILE = "expts/config_micro_ZINC.yaml"


def main(cfg: DictConfig) -> None:

    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)
    print("\ndatamodule:\n", datamodule, "\n")

    predictor = PredictorModule.load_from_checkpoint(MODEL_FILE)
    predictor.metrics = {}

    print(predictor.model)
    print(predictor.summarize(mode=4, to_print=False))

    trainer = load_trainer(cfg)

    # Run the model testing
    # trainer.test(model=predictor, datamodule=datamodule, ckpt_path=MODEL_FILE)
    
    predictions = trainer.predict(model=predictor, datamodule=datamodule)
    predictions = np.concatenate(predictions, axis=0)

    print(predictions)


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
