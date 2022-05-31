import os
import pickle
from logging import INFO

import numpy as np
import hashlib
from data_factory.preprocessing import *
from pytorch_forecasting import TimeSeriesDataSet
from utilities.config import load_config

logging.basicConfig(level=DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(DEBUG)


class StockPricesLoader:
    def __init__(self, config_file='../config/config.yml', log_level: int = INFO, use_previous_files=True, export=True):
        logger.setLevel(log_level)

        config = load_config(config_file)
        self.config = config
        self.export = export

        self.model = config['model']
        self.model_config = config[self.model]

        # Origin version (with basic fillna)
        self.df_train_origin, self.df_test_origin = None, None
        # As above but split among train val test
        self.df_train, self.df_val, self.df_test = None, None, None
        # Preprocessed version (scaled & fill missing dates)
        self.df_train_ppc, self.df_val_ppc, self.df_test_ppc = None, None, None
        # test df test + past dates for prediction
        self.df_test_ppc_ext = None
        # Timeseries datasets
        self.df_train_timeseries, self.df_val_timeseries, self.df_test_timeseries = None, None, None
        # Data loader of above timeseries
        self.train_dl, self.val_dl, self.test_dl = None, None, None

        # Scalers
        self.scalers = None

        # Create variables from config
        #  data loading
        self.save_folder = config['data']['save']
        self.train_file = config['data']['train_path'] + config['data']['stock_prices']
        self.test_file = config['data']['test_path'] + config['data']['stock_prices']
        #  TimeSeries settings
        self.max_prediction_length = self.model_config['sliding_window']['max_prediction_length']
        self.min_prediction_length = self.model_config['sliding_window']['min_prediction_length']
        self.max_encoder_length = self.model_config['sliding_window']['max_encoder_length']
        self.min_encoder_length = self.model_config['sliding_window']['min_encoder_length']
        self.batch_size = self.model_config['sliding_window']['batch_size']

        self.related_stocks = self.model_config['related_stock']
        self.train_val_split = self.model_config['train_val_split']
        self.scale = self.model_config['manual_scale']

        # define file name for saving StockPricesLoader with specific config
        hash_ = hashlib.md5(self.model_config.__str__().encode('utf-8')).hexdigest()
        self.export_file_name = f"{self.save_folder}/export_{hash_}.p"
        logger.debug(f'Export file {self.export_file_name}')

        logger.debug(f'Use config {config}')

        # Load data
        if use_previous_files and os.path.exists(self.export_file_name):
            logger.info(f"""Use previously generated file {self.export_file_name}. 
            Can not redo preprocessing by loading from generated file.""")
            self.from_file()
            return

        self.load()
        self.preprocess()
        self.add_related_stocks()
        self.to_timeseries()
        self.to_dataloader()
        self.to_file()

    def load(self):
        """
        Load data from origin files
        Change some types
        Parse dates
        Rename columns
        """
        self.df_train_origin = pd.read_csv(self.train_file, parse_dates=['Date'])
        logger.info(f'{self.train_file} loaded. shape {self.df_train_origin.shape}')

        self.df_test_origin = pd.read_csv(self.test_file, parse_dates=['Date'])
        logger.info(f'{self.test_file} loaded. shape {self.df_test_origin.shape}')

        for df in [self.df_test_origin, self.df_train_origin]:
            df['Timestamp'] = date_to_timestamp['1d'](df.Date.values.astype(np.int64)).astype(int)
            df.SupervisionFlag = df.SupervisionFlag.astype('category')
            df.SecuritiesCode = df.SecuritiesCode.astype(str)

    def preprocess(self):
        # Fill na
        logger.debug(f'Missing targets train {self.df_train_origin.Target.isna().sum()}')
        self.df_train_origin.dropna(subset=['Target'], inplace=True)
        logger.debug(f'Missing targets test {self.df_test_origin.Target.isna().sum()}')
        self.df_test_origin.dropna(subset=['Target'], inplace=True)
        logger.info('Missing targets dropped')

        self.df_train_origin.ExpectedDividend.fillna(value=0, inplace=True)
        self.df_train_origin.loc[:, ['Open', 'High', 'Low', 'Close']] = \
            self.df_train_origin.loc[:, ['Open', 'High', 'Low', 'Close']].fillna(method='ffill')

        self.df_test_origin.ExpectedDividend.fillna(value=0, inplace=True)
        self.df_test_origin.loc[:, ['Open', 'High', 'Low', 'Close']] = \
            self.df_test_origin.loc[:, ['Open', 'High', 'Low', 'Close']].fillna(method='ffill')
        logger.info(f'ExpectedDividend filled with 0. other values filled with ffill')

        if self.df_train_origin.isna().sum(axis=0).any():
            logger.warning('na values in train dataset')
        if self.df_test_origin.isna().sum(axis=0).any():
            logger.warning('na values in test dataset')

        self.df_train_origin.sort_values(by=['SecuritiesCode', 'Timestamp'], inplace=True)
        self.df_train_origin.reset_index(drop=True, inplace=True)
        self.df_test_origin.sort_values(by=['SecuritiesCode', 'Timestamp'], inplace=True)
        self.df_test_origin.reset_index(drop=True, inplace=True)

        self.df_train, self.df_val = split_train_val_timeseries(self.df_train_origin)
        self.df_test = self.df_test_origin.copy()

        self.df_train_ppc, self.df_val_ppc, self.df_test_ppc = self.df_train.copy(), self.df_val.copy(), self.df_test.copy()

        if self.scale:
            self._scale()

        self._fill_missing_dates()

        self._add_previous_timestamps_to_test_set()

        logger.info('Preprocessing not integrated in TimeSeriesDataSet done.')

    def _scale(self):
        self.scalers = train_scalers_on_timeseries(self.df_train)

        self.df_train_ppc = scale_timeseries(self.df_train_ppc, scalers=self.scalers)
        self.df_val_ppc = scale_timeseries(self.df_val_ppc, scalers=self.scalers)
        self.df_test_ppc = scale_timeseries(self.df_test_ppc, scalers=self.scalers)

        logger.debug('Data scaled')

    def _fill_missing_dates(self):
        self.df_train_ppc = fill_missing_dates(self.df_train_ppc, date_col='Date', timestamp_col='Timestamp',
                                               grp_col='SecuritiesCode', freq='1d',
                                               fill_with_value={'ExpectedDividend': 0})
        self.df_val_ppc = fill_missing_dates(self.df_val_ppc, date_col='Date', timestamp_col='Timestamp',
                                             grp_col='SecuritiesCode', freq='1d',
                                             fill_with_value={'ExpectedDividend': 0})
        self.df_test_ppc = fill_missing_dates(self.df_test_ppc, date_col='Date', timestamp_col='Timestamp',
                                              grp_col='SecuritiesCode', freq='1d',
                                              fill_with_value={'ExpectedDividend': 0})
        logger.debug('Fill missing dates done')

    def __dict__(self):
        return {
            'df_train_timeseries': self.df_train_timeseries,
            'df_val_timeseries': self.df_val_timeseries,
            'df_test_timeseries': self.df_test_timeseries,
            'train_dl': self.train_dl,
            'val_dl': self.val_dl,
            'test_dl': self.test_dl,
            'df_train_ppc': self.df_train_ppc,
            'df_val_ppc': self.df_val_ppc,
            'df_test_ppc': self.df_test_ppc,
            'scalers': self.scalers,
            'df_test_ppc_ext': self.df_test_ppc_ext
        }

    def to_file(self):
        """
        Export preprocessed data to a file whose name is the config.
        """
        if not self.export:
            return
        pickle.dump(self.__dict__(), open(self.export_file_name, "wb"))
        logger.info(f'Data exported to "{self.export_file_name}"')

    def from_file(self):
        """
        Load data from previously generated file
        This method of loading does not allow to redo the preprocessing part.
        """
        dt = pickle.load(open(self.export_file_name, "rb"))
        self.df_train_timeseries = dt['df_train_timeseries']
        self.df_test_timeseries = dt['df_test_timeseries']
        self.df_val_timeseries = dt['df_val_timeseries']
        self.train_dl = dt['train_dl']
        self.test_dl = dt['test_dl']
        self.val_dl = dt['val_dl']
        self.df_train_ppc = dt['df_train_ppc']
        self.df_val_ppc = dt['df_val_ppc']
        self.df_test_ppc = dt['df_test_ppc']
        self.scalers = dt['scalers']
        self.df_test_ppc_ext = dt['df_test_ppc_ext']

    def to_timeseries(self):
        additionals_cols = [f'Close_scaled_top_{i}' for i in range(self.related_stocks)]
        logger.info(f'Add columns {additionals_cols}')
        # Train timeseries
        self.df_train_timeseries = TimeSeriesDataSet(self.df_train_ppc,
                                                     time_idx='Timestamp',
                                                     target='Close_scaled',
                                                     group_ids=['SecuritiesCode'],
                                                     allow_missing_timesteps=False,
                                                     # static_categoricals=['SecuritiesCode'],
                                                     time_varying_unknown_reals=['Close_scaled'],
                                                     # time_varying_unknown_reals=['Open', 'High', 'Low', 'Close', 'Volume'],
                                                     # time_varying_unknown_categoricals=['SupervisionFlag'],
                                                     time_varying_known_reals=additionals_cols,

                                                     min_encoder_length=self.min_encoder_length,
                                                     max_encoder_length=self.max_encoder_length,
                                                     max_prediction_length=self.max_prediction_length,
                                                     min_prediction_length=self.min_prediction_length,
                                                     scalers={col: None for col in ['Open', 'High', 'Low', 'Volume', 'AdjustmentFactor', 'ExpectedDividend'] + additionals_cols},
                                                     target_normalizer=None,
                                                     add_relative_time_idx=False,
                                                     add_target_scales=False,
                                                     add_encoder_length=False,
                                                     )
        logger.debug('train timeseries created')

        # Validation timeseries
        self.df_val_timeseries = None
        if self.train_val_split != 1:
            self.df_val_timeseries = TimeSeriesDataSet.from_dataset(self.df_train_timeseries, self.df_val_ppc,
                                                                    predict=False,
                                                                    stop_randomization=True)
            logger.debug('validation timeseries created')

        # Test timeseries
        self.df_test_timeseries = TimeSeriesDataSet.from_dataset(
            self.df_train_timeseries, self.df_test_ppc_ext,
            allow_missing_timesteps=False,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=self.df_test_origin.Timestamp.min() + 1,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length
        )
        logger.debug('test timeseries created')
        logger.info('Timeseries created')

    def to_dataloader(self):
        """
        Create data loaders for model
        """
        self.train_dl = self.df_train_timeseries.to_dataloader(train=True, batch_size=self.batch_size, num_workers=12)
        logger.debug('train data loader created')

        self.val_dl = None
        if self.df_val_timeseries:
            self.val_dl = self.df_val_timeseries.to_dataloader(train=False, batch_size=self.batch_size, num_workers=12)
        logger.debug('validation data loader created')

        self.test_dl = self.df_test_timeseries.to_dataloader(
            batch_size=self.df_test_origin.Timestamp.max() - self.df_test_origin.Timestamp.min() + 1,
            num_workers=12, shuffle=False)
        logger.debug('test data loader created')
        logger.info('Data Loaders created')

    def _add_previous_timestamps_to_test_set(self):
        #  Extend days so that TimeSeriesDataSet iterate over the last actual item in test dataset.
        self.df_test_ppc_ext = add_days(
            self.df_test_ppc,
            days=self.max_prediction_length,
            grp_col='SecuritiesCode',
            timestamp_col='Timestamp',
            date_col='Date'
        )
        #  Add previous dates to make use of them in the prediction
        self.df_test_ppc_ext = pd.concat([self.df_train_ppc, self.df_test_ppc_ext]).sort_values(
            by=['SecuritiesCode', 'Timestamp']).reset_index(drop=True)

        #  Fill missing dates in test dataframe so that the behaviour of TimeSeriesDataSet is always the same.
        self.df_test_ppc_ext = fill_missing_dates(self.df_test_ppc_ext, 'Date', 'Timestamp', 'SecuritiesCode', '1d',
                                                  fill_with_value={'ExpectedDividend': 0})

    def add_related_stocks(self):
        n = self.related_stocks
        if not n:
            return

        cosine = pd.read_csv(self.config['data']['suppl'] + self.config['data']['cosine'])
        cosine.set_index('ticker', inplace=True)
        self.top = cosine.apply(lambda s: pd.Series(s.nlargest(3).index)).T.astype(str).rename(columns=str)
        missing_keys = list(set(self.df_train_ppc.SecuritiesCode.unique()) - set(cosine.columns))
        logger.info(f"Len missing Securities Code in {self.config['data']['cosine']}: {len(missing_keys)} ")

        self.top = pd.concat([self.top, pd.DataFrame({str(i): missing_keys for i in range(n)}, index=missing_keys)])

        def _add_stocks(df: pd.DataFrame):
            for t, col in [(f'top_{i}', str(i)) for i in range(n)]:
                df[t] = df.SecuritiesCode.transform(lambda x: self.top.loc[x, col])
                df = df.merge(df.loc[:, ['SecuritiesCode', 'Timestamp', 'Close', 'Close_scaled']],
                              how='left',
                              left_on=[t, 'Timestamp'],
                              right_on=['SecuritiesCode', 'Timestamp'],
                              suffixes=('', f'_{t}')).drop(columns=f'SecuritiesCode_{t}')
                df[f'Close_scaled_{t}'] = df[f'Close_scaled_{t}'].shift(self.max_prediction_length)
            return df.fillna(value=0)

        self.df_train_ppc = _add_stocks(self.df_train_ppc)
        self.df_test_ppc = _add_stocks(self.df_test_ppc)
        self.df_test_ppc_ext = _add_stocks(self.df_test_ppc_ext)
        self.df_val_ppc = _add_stocks(self.df_val_ppc)


if __name__ == '__main__':
    dl = StockPricesLoader(log_level=DEBUG, use_previous_files=True)
    print(len(dl.train_dl))
    print(len(dl.test_dl))
    print(len(dl.val_dl))
