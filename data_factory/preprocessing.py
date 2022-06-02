from __future__ import annotations

import logging
from logging import DEBUG
from typing import Literal

import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(DEBUG)

date_to_timestamp = {
    '1min': lambda date_val: date_val / 10 ** 9,
    '1d': lambda date_val: date_val / 10 ** 9 / (24 * 60 * 60)
}


def fill_missing_dates(df: pd.DataFrame,
                       date_col: str,
                       timestamp_col: str,
                       grp_col: str,
                       freq: str = '1min',
                       main_method: Literal["backfill", "bfill", "ffill", "pad"] = 'ffill',
                       fill_with_value: dict[str, any] | None = None,
                       fill_with_method: dict[str, Literal["backfill", "bfill", "ffill", "pad"]] | None = None
                       ) -> pd.DataFrame:

    def fill_na(data: pd.DataFrame):
        # logger.debug(data.columns)
        logger.debug(f'Fill grp {data[grp_col].unique()[0]}. shape: {data.shape}')
        start = data[date_col].min()
        end = data[date_col].max()
        idx = pd.date_range(start=start, end=end, freq=freq)
        data.set_index(date_col, inplace=True)
        data = data.reindex(idx)

        if freq not in date_to_timestamp:
            raise IndexError(f"""
            freq {freq} not found in date_to_timestamp transformations.
            Possible values are {list(date_to_timestamp.keys())}
            """)

        data[timestamp_col] = date_to_timestamp[freq](data.index.astype(int)).astype(int)
        data[grp_col] = data[grp_col].unique()[0]
        data.authentic.fillna(value=False, inplace=True)

        if fill_with_method is not None:
            for col in fill_with_method:
                data[col].fillna(method=fill_with_method[col], inplace=True)
        if fill_with_value is not None:
            for col in fill_with_value:
                data[col].fillna(value=fill_with_value[col], inplace=True)
        data.fillna(method=main_method, inplace=True)
        logger.debug(f'Fill grp {data[grp_col].unique()[0]} Done. shape: {data.shape}')
        return data.reset_index(drop=False).rename(columns={'index': date_col})

    if 'authentic' not in df.columns:
        df['authentic'] = True
    df = df.groupby(grp_col).apply(fill_na).reset_index(drop=True)
    return df


def add_days(df: pd.DataFrame,
             days: int,
             grp_col: str,
             timestamp_col: str,
             date_col: str):
    """
    Add x days for each grp_col. Can be useful when using TimeSeriesDataset to
    make it output values until the (last_one - min_prediction_length)
    :param df: dataset
    :param days: number of days
    :param grp_col: column to group over
    :param timestamp_col: column name containing the timestamp (used to add days)
    :param date_col: date column name, updated after adding new timestamps
    :return: dataset with new days
    """

    logger.warning('This function is not ready to be used with something else than test set')

    def _add_days(data: pd.DataFrame):
        a = data.loc[:, grp_col].iloc[0]
        data = data.copy().set_index(timestamp_col, drop=True)
        data = data.reindex(data.index.to_list() + list(range(data.index.max() + 1, data.index.max() + days + 1)))
        data.reset_index(drop=False, inplace=True)
        data.loc[:, grp_col] = a
        data.authentic.fillna(value=False, inplace=True)
        data.loc[data.AdjustmentFactor.isnull(), 'AdjustmentFactor'] = 1.  # Not generic yet. TODO
        data.loc[data.AdjustmentFactor.isnull(), 'is_testing'] = False  # Not generic yet. TODO
        data.loc[:, date_col] = pd.to_datetime(data.Timestamp,
                                               unit='d')  # make it possible to add smthg else than days TODO
        data.fillna(0, inplace=True)  # make this configurable TODO
        return data

    if 'authentic' not in df.columns:
        df['authentic'] = True
    df_test_ext = df.groupby(grp_col).apply(_add_days).reset_index(drop=True)
    return df_test_ext


def split_train_val_timeseries(df: pd.DataFrame,
                               grp_col='SecuritiesCode',
                               train_val_split=.95) -> (pd.DataFrame, pd.DataFrame):

    train = df.groupby(grp_col).apply(lambda x: x.iloc[:round(x.shape[0] * train_val_split)]).reset_index(drop=True)
    val = df.groupby(grp_col).apply(lambda x: x.iloc[round(x.shape[0] * train_val_split):]).reset_index(drop=True)
    return train, val


def train_scalers_on_timeseries(df: pd.DataFrame,
                                col='Close',
                                grp_col='SecuritiesCode',
                                scaler=StandardScaler) -> pd.Series:

    return df.groupby(grp_col)[col].apply(lambda x: scaler().fit(x.to_numpy().reshape(-1, 1))).rename('scaler')


def scale_timeseries(df: pd.DataFrame,
                     scalers: pd.Series,
                     col='Close',
                     grp_col='SecuritiesCode') -> pd.DataFrame:

    df[f'{col}_scaled'] = df.groupby(grp_col)[col].transform(lambda x: scalers[x.name].transform(x.to_numpy().reshape(-1, 1)).reshape(-1))
    return df


def unscale_timeseries(df: pd.DataFrame,
                       scalers: pd.Series,
                       col='Close',
                       grp_col='SecuritiesCode') -> pd.DataFrame:

    df[f'{col}_unscaled'] = df.groupby(grp_col)[col].transform(lambda x: scalers[x.name].inverse_transform(x.to_numpy().reshape(-1, 1)).reshape(-1))
    return df
