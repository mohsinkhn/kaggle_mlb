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

    def __init__(self, date_col, user_col, key_cols, hist_data_path, player_mapping, fill_value=-1, N=1000):
        """Initialization."""
        self.date_col = date_col
        self.user_col = user_col
        self.key_cols = key_cols
        self.hist_data_path = hist_data_path
        self.player_mapping = load_json(player_mapping)
        self.cols = [self.date_col, self.user_col]
        self.FILL_VALUE = fill_value
        self.N = N

    def _transform(self, X):
        # Load past data for aggregation and convert to cupy array
        data, date_mapping = self._load_historical_data(self.hist_data_path)
        data = cp.array(data)
        tdim, udim, cdim = *data.shape[:2], len(self.key_cols)
        # loop over dates and aggregate data for each date
        dates = X[self.date_col].unique()
        date_to_idx = {}
        results = []
        idx = 0
        max_date = max(date_mapping.keys())
        for i, date in enumerate(dates):
            date = str(date)
            if date > max_date:
                idx = tdim
            else:
                if date in date_mapping:
                    idx = date_mapping[date]

            agg = self.agg_date_data(data, idx, udim, cdim)
            results.append(agg)
            date_to_idx[date] = i
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
            subdata = data_cp[:idx+1, :, self.key_cols]
            agg = self._reduce_func(subdata)
        return agg

    @abstractmethod
    def _reduce_func(self, arr):
        return cp.nanmean(arr, axis=0)


class ExpandingMax(TimeSeriesTransformer):
    """Expanding max based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanmax(arr, axis=0)


class ExpandingMean(TimeSeriesTransformer):
    """Expanding mean based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanmean(arr[:-1], axis=0)


class ExpandingMedian(TimeSeriesTransformer):
    """Expanding median based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanmedian(arr[:-1], axis=0)


class ExpandingQ05(TimeSeriesTransformer):
    """Expanding 5th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanquantile(arr, 0.05, axis=0)


class ExpandingQ25(TimeSeriesTransformer):
    """Expanding 25th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanquantile(arr, 0.25, axis=0)


class ExpandingQ75(TimeSeriesTransformer):
    """Expanding 75th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanquantile(arr, 0.75, axis=0)


class ExpandingQ95(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.nanquantile(arr, 0.95, axis=0)


class LagN(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return arr[0]
        return arr[-self.N]


class LastNMean(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return cp.nanmean(arr, 0)
        return cp.nanmean(arr[-self.N:], 0)


class LastNMedian(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        if len(arr) < self.N:
            return cp.nanmedian(arr, 0)
        return cp.nanmedian(arr[-self.N:], 0)


class OrdinalTransformer(BaseTransformer):
    """Encode some of the columns from dataframe as ordinal values."""
    def __init__(self, cols):
        self.cols = cols
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        self.enc.fit(X[self.cols].astype(str))
        return self

    def transform(self, X, y=None):
        Xord = self.enc.transform(X[self.cols].astype(str))
        other_cols = [col for col in X.columns if col not in self.cols]
        df = X[other_cols]
        for i, col in enumerate(self.cols):
            df[col] = Xord[:, i]
        return df


class DateTimeFeatures(ArrayTransformer):
    def __init__(self, attrs=['dayofweek', 'day'], format='%Y%m%d'):
        self.attrs = attrs
        self.format = format

    def _transform(self, X, y=None):
        X = pd.to_datetime(X, format=self.format)
        Xt = []
        for attr in self.attrs:
            out = getattr(X.dt, attr).values
            Xt.append(out)
        return np.vstack(Xt).T
