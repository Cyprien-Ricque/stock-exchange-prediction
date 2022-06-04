import sys

sys.path.append("../../pytorch-forecasting")
sys.path.append("../")

import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import notebook

from utilities import *
import pandas as pd
from tqdm import tqdm

from data_factory.prepared_data import TimeSeriesData, PreparedData

sns.set_style("whitegrid")
tqdm.pandas()

pl.seed_everything(42)

config = load_config("../config/config.yml")
assert config['model'] == 'temporal_fusion_transformer', 'Invalid model in file configuration for this script'
model = config['model']

data_ts: TimeSeriesData = TimeSeriesData.from_file('../data/save/timeseries_data.pkl')

batch_size = config[model]['sliding_window']['batch_size']

# Training
train_dl = data_ts.train.to_dataloader(train=True, batch_size=batch_size, num_workers=12)


# Validation
val_dl = data_ts.val.to_dataloader(train=False, batch_size=batch_size, num_workers=12, shuffle=False)


# Testing
test_dl = data_ts.test.to_dataloader(
    batch_size=data_ts.test_set_size,
    num_workers=12,
    shuffle=False
)

from pytorch_forecasting.models import TemporalFusionTransformer
import logging
from logging import WARNING
logging.basicConfig(level=WARNING)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=2, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")


args = dict(
    hidden_size=config['temporal_fusion_transformer']['hidden_size'],
    lstm_layers=config['temporal_fusion_transformer']['lstm_layers'],
    dropout=config['temporal_fusion_transformer']['dropout'],
    attention_head_size=config['temporal_fusion_transformer']['attention_head_size'],
    output_size=config['temporal_fusion_transformer']['output_size'],
)

trainer = pl.Trainer(
    accelerator='gpu',
    gradient_clip_val=0.1,
    auto_lr_find=True,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
    weights_summary="top",
    max_epochs=10
)

model = TemporalFusionTransformer.from_dataset(
    data_ts.train,
    **args
)

# model = TemporalFusionTransformer.load_from_checkpoint('./lightning_logs/lightning_logs/version_41/checkpoints/epoch=0-step=46790.ckpt')

print(f"Number of parameters in network: {model.size() / 1e3:.1f}k")


trainer.validate(model, val_dl)


trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


trainer.validate(model, val_dl)