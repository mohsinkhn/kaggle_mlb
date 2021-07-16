"""All the transformers go here."""

from abc import abstractmethod

import cupy as cp
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from mllib.utils import load_json


def _convert_to_2d_array(X):
    X = np.array(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base interface for transformer."""

    def fit(self, X, y=None):
        """Learn something from data."""
        return self

    @abstractmethod
    def _transform(self, X):
        pass

    def transform(self, X, y=None):
        """Transform data."""
        return self._transform(X)


class ArrayTransformer(BaseTransformer):
    """Transformer to be used for returnng 2d arrays."""

    def transform(self, X):
        """Transform data and return 2d array."""
        Xt = self._transform(X)
        return _convert_to_2d_array(Xt)


class SelectCols(BaseTransformer):
    """Select column of a dataframe."""

    def __init__(self, cols):
        """Initialie columns to be selected."""
        self.cols = cols

    def _transform(self, X, y=None):
        return X[self.cols]


class FunctionTransfomer(ArrayTransformer):
    """Apply a func on arrays which returns back arrays."""

    def __init__(self, func):
        """Initialize function to be used."""
        self.func = func

    def _transform(self, X):
        return self.func(X)


class TimeSeriesTransformer(ArrayTransformer):
    """Expanding operations on historical artifcats."""

    def __init__(self, date_col, user_col, key_cols, hist_data_path, player_mapping, fill_value=-1, N=1000, skip=0):
        """Initialization."""
        self.date_col = date_col
        self.user_col = user_col
        self.key_cols = key_cols
        self.hist_data_path = hist_data_path
        self.player_mapping = load_json(player_mapping)
        self.cols = [self.date_col, self.user_col]
        self.FILL_VALUE = fill_value
        self.N = N
        self.skip = skip

    def _transform(self, X):
        # Load past data for aggregation and convert to cupy array
        data, date_mapping = self._load_historical_data(self.hist_data_path)
        data = cp.array(data)
        _, udim, cdim = *data.shape[:2], len(self.key_cols)
        # loop over dates and aggregate data for each date
        dates = X[self.date_col].unique()
        date_to_idx = {}
        results = []
        idx = 0
        dates_int = np.sort([int(self._shift(d)) for d in date_mapping.keys()])
        indices = np.searchsorted(dates_int, dates, side='right')
        for i, idx in enumerate(indices):
            agg = self.agg_date_data(data, idx, udim, cdim)
            results.append(agg)
            date_to_idx[str(dates[i])] = i
        out = cp.stack(results).get()

        # map to input dates, playerIds
        Xt = []
        for dt, pid in X[self.cols].values:
            if str(pid) in self.player_mapping:
                Xt.append(out[date_to_idx[str(dt)], self.player_mapping[str(pid)]])
            else:
                Xt.append(np.ones(shape=(out.shape[2],)) * self.FILL_VALUE)
        return np.array(Xt)

    def _load_historical_data(self, filepath):
        data = np.load(str(Path(filepath) / 'data.npy')).astype(np.float32)
        date_mapping = load_json(str(Path(filepath) / 'date_mapping.json'))
        return data, date_mapping

    def agg_date_data(self, data_cp, idx, udim, cdim):
        if idx == 0:
            agg = cp.ones(shape=(udim, cdim), dtype=np.float32) * self.FILL_VALUE
        else:
            subdata = data_cp[:idx, :, self.key_cols]
            agg = self._reduce_func(subdata)
        return agg

    @abstractmethod
    def _reduce_func(self, arr):
        return cp.nanmean(arr, axis=0)

    def _shift(self, date):
        date = pd.to_datetime(date, format="%Y%m%d") - pd.Timedelta(self.skip)
        return f"{date:%Y%m%d}"


class ExpandingMax(TimeSeriesTransformer):
    """Expanding max based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanmax(arr, axis=0)


class ExpandingSum(TimeSeriesTransformer):
    """Expanding max based on historical data."""

    def _reduce_func(self, arr):
        return cp.nansum(arr, axis=0)


class ExpandingVar(TimeSeriesTransformer):
    """Expanding max based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanvar(arr, axis=0)


class ExpandingMin(TimeSeriesTransformer):
    """Expanding max based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanmin(arr, axis=0)


class ExpandingMean(TimeSeriesTransformer):
    """Expanding mean based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanmean(arr, axis=0)


class ExpandingMedian(TimeSeriesTransformer):
    """Expanding median based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanmedian(arr, axis=0)


class ExpandingQ05(TimeSeriesTransformer):
    """Expanding 5th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.quantile(arr, 0.05, axis=0)


class ExpandingQ25(TimeSeriesTransformer):
    """Expanding 25th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.quantile(arr, 0.25, axis=0)


class ExpandingQ75(TimeSeriesTransformer):
    """Expanding 75th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.quantile(arr, 0.75, axis=0)


class ExpandingQ95(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.quantile(arr, 0.95, axis=0)


class LagN(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return arr[0]
        subarr = arr[-self.N]
        subarr[cp.isnan(subarr)] = self.FILL_VALUE
        return subarr


class UnqLastN(TimeSeriesTransformer):
    """Unique values in last N"""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return arr[0]
        subarr = arr[-self.N:]
        subarr[cp.isnan(subarr)] = self.FILL_VALUE
        subarr = cp.sum(subarr[1:] != subarr[:-1], 0)
        # subarr[cp.isnan(subarr)] = self.FILL_VALUE
        return subarr


class LastNMean(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return arr[0]
        subarr = arr[-self.N:]
        return cp.nanmean(subarr, 0)


class LastNMedian(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return arr[0]
        subarr = arr[-self.N:]
        return cp.nanmean(subarr, 0)


class LastNSum(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return arr[0]
        subarr = arr[-self.N:]
        return cp.nansum(subarr, 0)


class OrdinalTransformer(BaseTransformer):
    """Encode some of the columns from dataframe as ordinal values."""
    def __init__(self, cols):
        self.cols = cols
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        self.enc.fit(X[self.cols].astype(str))
        return self

    def transform(self, X, y=None):
        if X is None:
            return None
        if not isinstance(X, pd.DataFrame):
            return None
        if (len(set(self.cols) - set(X.columns)) == 0):
            Xord = self.enc.transform(X[self.cols].astype(str))
            df = pd.DataFrame(Xord, index=X.index, columns=self.cols)
            other_cols = [col for col in X.columns if col not in self.cols]
            if len(other_cols) > 0:
                for col in other_cols:
                    df[col] = X[col]
            return df
        else:
            return None


class DateTimeFeatures(ArrayTransformer):
    def __init__(self, attrs=['dayofweek', 'day'], format='%Y%m%d'):
        self.attrs = attrs
        self.format = format

    def _transform(self, X, y=None):
        if X is None:
            return None
        try:
            X = pd.to_datetime(X, format=self.format)
        except AttributeError:
            return None
        Xt = []
        for attr in self.attrs:
            out = getattr(X.dt, attr).values
            Xt.append(out)
        return np.vstack(Xt).T


class MapAttributes(ArrayTransformer):
    def __init__(self, attr_file='data/players.csv', file_type='csv', map_col='playerId', attr='playerPositionCode'):
        self.attr_file = attr_file
        self.attr = attr
        self.map_col = map_col
        self.file_type = file_type
        self.setup()

    def setup(self):
        if self.file_type == 'csv':
            self.data = pd.read_csv(self.attr_file)
            self.data = self.data[[self.map_col, self.attr]].drop_duplicates()
            self.data = self.data.set_index(self.map_col)[self.attr].to_dict()
        else:
            self.data = load_json(self.attr_file)[self.map_col]

    def _transform(self, X, y=None):
        if X is None:
            return None
        if not isinstance(X, pd.DataFrame):
            return None
        if self.map_col not in X.columns:
            return None
        return X[self.map_col].map(self.data)


class Astype(ArrayTransformer):
    def __init__(self, totype='int32', values_to_map=None):
        self.totype = totype
        self.values_to_map = values_to_map

    def _transform(self, X, y=None):
        if X is None:
            return None

        Xt = []
        for row in X:
            if row[0] in self.values_to_map:
                Xt.append([self.values_to_map[row[0]]])
            else:
                Xt.append([row[0]])
        return np.array(Xt).astype(self.totype)


class DateDiff(ArrayTransformer):
    def __init__(self, date_col='date', user_col='playerId', diff_col='mlbDebutDate', data_filepath='data/players.csv', format='%Y%m%d'):
        self.date_col = date_col
        self.user_col = user_col
        self.diff_col = diff_col
        self.format = format
        self.data_filepath = data_filepath
        self.setup()

    def setup(self):
        data = pd.read_csv(self.data_filepath)
        data[self.diff_col] = pd.to_datetime(data[self.diff_col])
        data = data[[self.user_col, self.diff_col]].drop_duplicates()
        self.diffdate = data.set_index(self.user_col)[self.diff_col].to_dict()

    def _transform(self, X, y=None):
        if X is None:
            return None
        Xdt = X[self.user_col].map(self.diffdate)
        return (pd.to_datetime(X[self.date_col], format=self.format) - Xdt).dt.days


class MapSeasonFlag(ArrayTransformer):
    def __init__(self, date_col='date', season_file="data/seasons.csv"):
        self.date_col = date_col
        self.season_file = season_file
        self.setup()

    def setup(self):
        seasons = pd.read_csv(self.season_file)
    
    def _transform(self, X, y=None):
        Xdt = pd.to_datetime(X[self.date_col], format="%Y%m%d")
        years = Xdt.dt.year