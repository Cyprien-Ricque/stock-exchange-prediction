#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append("../../pytorch-forecasting")
sys.path.append("../")
# sys.path.append("/media/cyprien/Data/Documents/Github/pytorch-forecasting")

import pytorch_lightning as pl
import seaborn as sns
import torch
from data_factory.dataLoader import StockPricesLoader
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline
from utilities import *

from tqdm import tqdm

sns.set_style("whitegrid")

pl.seed_everything(42)


# In[2]:


config = load_config("../config/config.yml")
config


# In[3]:


dl = StockPricesLoader(use_previous_files=True)


# ## Baseline Model

# ### Test set

# In[4]:


actuals = torch.cat([y for x, (y, weight) in tqdm(iter(dl.test_dl))])
baseline_predictions = Baseline().predict(dl.test_dl)
print((actuals - baseline_predictions).abs().mean().item())

baseline_predictions_np = baseline_predictions.cpu().detach().numpy()
actuals_np = actuals.cpu().detach().numpy()


# In[5]:


df_test_res_baseline = dl.df_test_ppc.copy()

df_test_res_baseline['close_true'] = actuals_np[:, 0].flatten()
df_test_res_baseline['close_pred'] = baseline_predictions_np[:, 0].flatten()
df_test_res_baseline['target_true'] = (actuals_np[:, 1] - actuals_np[:, 0]) / actuals_np[:, 0]
df_test_res_baseline['target_pred'] = (baseline_predictions_np[:, 1] - baseline_predictions_np[:, 0]) / baseline_predictions_np[:, 0]
df_test_res_baseline.loc[:, ['Close', 'Close_scaled', 'close_true', 'close_pred', 'Target', 'target_true', 'target_pred']]


# ## RNN

# In[6]:


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


# #### Train the model

# In[7]:


# # fit network

fit = True

if fit:
    trainer.fit(rnn, train_dataloaders=dl.train_dl, val_dataloaders=dl.val_dl)


# ### Results

# In[8]:


import numpy as np

actuals = torch.cat([y for _, (y, _) in tqdm(iter(dl.test_dl))])
predictions = trainer.predict(rnn, (X for X, (y, _) in dl.test_dl))

predictions_np = np.array([i.prediction.numpy() for i in predictions]).squeeze(axis=3).reshape(-1, 10)
actuals_np = actuals.cpu().detach().numpy()


# In[9]:


actuals_unscaled = actuals_np.reshape(dl.df_train_ppc.SecuritiesCode.unique().size, -1, 10).copy()
predictions_unscaled = predictions_np.reshape(dl.df_train_ppc.SecuritiesCode.unique().size, -1, 10).copy()

for i, scaler in enumerate(dl.scalers):
    actuals_unscaled[i] = scaler.inverse_transform(actuals_unscaled[i])
for i, scaler in enumerate(dl.scalers):
    predictions_unscaled[i] = scaler.inverse_transform(predictions_unscaled[i])

actuals_unscaled = actuals_unscaled.reshape(-1, 10)
predictions_unscaled = predictions_unscaled.reshape(-1, 10)


# In[10]:


df_test_res = dl.df_test_ppc.copy()

df_test_res['close_true'] = actuals_np[:, 0].flatten()
df_test_res['close_pred'] = predictions_np[:, 0].flatten()

df_test_res['close_true_unscaled'] = actuals_unscaled[:, 0].flatten()
df_test_res['close_pred_unscaled'] = predictions_unscaled[:, 0].flatten()

df_test_res['target_true'] = (actuals_np[:, 1] - actuals_np[:, 0]) / actuals_np[:, 0]
df_test_res['target_pred'] = (predictions_np[:, 1] - predictions_np[:, 0]) / predictions_np[:, 0]

df_test_res['target_true_unscaled'] = (actuals_unscaled[:, 1] - actuals_unscaled[:, 0]) / actuals_unscaled[:, 0]
df_test_res['target_pred_unscaled'] = (predictions_unscaled[:, 1] - predictions_unscaled[:, 0]) / predictions_unscaled[:, 0]

df_test_res.loc[:, ['Close', 'close_true_unscaled', 'close_pred_unscaled', 'Target', 'target_true_unscaled', 'target_pred_unscaled']]


# In[18]:


import matplotlib.pyplot as plt

for sc in dl.df_train_ppc.SecuritiesCode.unique()[:3]:
    df = df_test_res[df_test_res.SecuritiesCode == sc]
    figure = plt.figure(figsize=(20, 5))
    # plt.plot(df.Date, df.Close, label='close true', figure=figure)
    plt.plot(df.Date, df.close_true_unscaled, label='close true', figure=figure)
    plt.plot(df.Date, df.close_pred_unscaled, label='close pred', figure=figure)

    plt.legend()
    plt.show()


# In[13]:


import matplotlib.pyplot as plt

recreated_target_is_valid = df_test_res.groupby('SecuritiesCode').apply(lambda x: (x.authentic == True).shift(-2).fillna(value=False) & (x.authentic == True).shift(-1).fillna(value=False) & (x.authentic == True)).reset_index(drop=True)
evaluated_target = (df_test_res.authentic == True)

for sc in df_test_res.SecuritiesCode.unique()[:8]:
    df = df_test_res[(df_test_res.SecuritiesCode == sc) & evaluated_target]
    figure = plt.figure(figsize=(20, 5))
    plt.scatter(df.Date, df.Target, label='true', figure=figure, alpha=.5)
    # plt.scatter(df.Date, df.target_true_unscaled, label='true2', figure=figure, alpha=.5)
    plt.scatter(df.Date, df.target_pred_unscaled, label='pred', figure=figure, alpha=.5)
    plt.legend()
    plt.show()


# In[14]:


from utilities.evaluation import calc_spread_return_sharpe


# In[15]:


df_test_res['Rank'] = (df_test_res.groupby("Date")["target_true_unscaled"].rank(ascending=False, method="first") - 1).astype(int)
calc_spread_return_sharpe(df_test_res)


# In[16]:


df_test_res['Rank'] = (df_test_res.groupby("Date")["target_pred_unscaled"].rank(ascending=False, method="first") - 1).astype(int)
calc_spread_return_sharpe(df_test_res)


# In[ ]:




