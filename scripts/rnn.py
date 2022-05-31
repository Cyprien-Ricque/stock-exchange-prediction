import sys

sys.path.append("../../pytorch-forecasting")
sys.path.append("../")

import pytorch_lightning as pl
import seaborn as sns
import torch
from data_factory.dataLoader import StockPricesLoader
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import notebook

from pytorch_forecasting import Baseline
from utilities import *


notebook.tqdm().pandas()

pl.seed_everything(42)


config = load_config("../config/config.yml")
assert config['model'] == 'rnn', 'Invalid model in file configuration for this script'
dl = StockPricesLoader(use_previous_files=True, export=False)


from pytorch_forecasting.models.rnn import RecurrentNetwork

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

args = dict(
    hidden_size=config['rnn']['hidden_size'],
    rnn_layers=config['rnn']['layers'],
    dropout=config['rnn']['dropout']
)

# configure network and trainer
trainer = pl.Trainer(
    accelerator='gpu',
    gradient_clip_val=0.1,
    # clipping gradients is a hyperparameter and important to prevent divergence
    # of the gradient for recurrent neural networks
    auto_lr_find=True,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
    weights_summary="top",
)

rnn = RecurrentNetwork.from_dataset(
    dl.df_train_timeseries,
    **args
)

# rnn = RecurrentNetwork.load_from_checkpoint('././lightning_logs/lightning_logs/version_5/checkpoints/epoch=0-step=49915.ckpt')

print(f"Number of parameters in network: {rnn.size() / 1e3:.1f}k")


# # fit network

trainer.fit(rnn, train_dataloaders=dl.train_dl, val_dataloaders=dl.test_dl)