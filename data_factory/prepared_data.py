from __future__ import annotations
import sys
sys.path.append("../../pytorch-forecasting")

from pytorch_forecasting import TimeSeriesDataSet

from dataclasses import dataclass
import pandas as pd
import pickle


@dataclass
class PreparedData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scalers: dict[str | pd.Series]

    def export(self, file='prepared_data.pkl'):
        pickle.dump(self, open(file, 'wb'))

    @classmethod
    def from_file(cls, file='prepared_data.pkl'):
        return pickle.load(open(file, 'rb'))


@dataclass
class TimeSeriesData:
    train: TimeSeriesDataSet
    val: TimeSeriesDataSet
    test: TimeSeriesDataSet

    # Other needed information for training
    test_set_size: int

    def export(self, file='timeseries_data.pkl'):
        pickle.dump(self, open(file, 'wb'))

    @classmethod
    def from_file(cls, file='timeseries_data.pkl'):
        return pickle.load(open(file, 'rb'))
