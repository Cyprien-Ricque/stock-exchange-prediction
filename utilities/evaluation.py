import numpy as np
import pandas as pd

import torch

# From Kaggle https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition/notebook


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """

    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1

        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


def sharp_ratio_loss(y_hat, targets):
    """

    :param y_hat: shape (day, stock price rank)
    :return:
    """
    def per_day(y_hat_day, targets_day, toprank_weight_ratio, portfolio_size):
        weights = torch.linspace(start=toprank_weight_ratio, end=1, steps=portfolio_size) + torch.zeros(2000 - portfolio_size)
        sorted_indices = torch.sort(y_hat_day)[1]

        target_derived_from_rank = y_hat_day * (targets_day / y_hat_day)
        torch.sort(target_derived_from_rank).based_on(y_hat_day)