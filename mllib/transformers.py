"""All the transformers go here."""

from abc import abstractmethod
import json

import cupy as cp
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


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

    def __init__(self, date_col, user_col, key_cols, hist_data_path, fill_value=-1, N=1000):
        """Initialization."""
        self.date_col = date_col
        self.user_col = user_col
        self.key_cols = key_cols
        self.cols = [self.date_col, self.user_col]
        self.FILL_VALUE = fill_value
        self.N = N
        self.data, self.player_mapping, self.date_mapping = self._load_historical_data(hist_data_path)
        self.data = cp.array(self.data)

    def _transform(self, X):
        # Load past data for aggregation and convert to cupy array
        tdim, udim, cdim = *self.data.shape[:2], len(self.key_cols)
        max_date = max(self.date_mapping.keys())
        # loop over dates and aggregate data for each date
        dates = X[self.date_col].unique()
        date_to_idx = {}
        results = []
        for i, date in enumerate(dates):
            date = str(date)
            if date > max_date:
                idx = tdim
            else:
                idx = self.date_mapping[date]
            agg = self.agg_date_data(self.data, idx, udim, cdim)
            results.append(agg)
            date_to_idx[date] = i
        out = cp.stack(results).get()

        # map to input dates, playerIds
        Xt = []
        for dt, pid in X[self.cols].values:
            Xt.append(out[date_to_idx[str(dt)], self.player_mapping[str(pid)]])
        return np.array(Xt)

    def _update(self, X):
        self.data = self.data.append(X)

    def _load_historical_data(self, filepath):
        data = np.load(str(Path(filepath) / 'data.npy')).astype(np.float32)
        player_mapping = self._load_dict(str(Path(filepath) / 'player_mapping.json'))
        date_mapping = self._load_dict(str(Path(filepath) / 'date_mapping.json'))
        return data, player_mapping, date_mapping

    def _load_dict(self, fp):
        with open(fp, "r") as f:
            out = json.load(f)
        return out

    def agg_date_data(self, data_cp, idx, udim, cdim):
        if idx == 0:
            agg = cp.ones(shape=(udim, cdim), dtype=np.float32) * self.FILL_VALUE
        else:
            subdata = data_cp[:idx, :, self.key_cols]
            agg = self._reduce_func(subdata)
        return agg

    @abstractmethod
    def _reduce_func(self, arr):
        return cp.mean(arr, axis=0)


class ExpandingMax(TimeSeriesTransformer):
    """Expanding max based on historical data."""

    def _reduce_func(self, arr):
        return cp.max(arr, axis=0)


class ExpandingMean(TimeSeriesTransformer):
    """Expanding mean based on historical data."""

    def _reduce_func(self, arr):
        return cp.mean(arr, axis=0)


class ExpandingMedian(TimeSeriesTransformer):
    """Expanding median based on historical data."""

    def _reduce_func(self, arr):
        return cp.median(arr, axis=0)


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
        return arr[-self.N]


class LastNMean(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.mean(arr[-self.N:], 0)


class LastNMedian(TimeSeriesTransformer):
    """Expanding 95th percentile based on historical data."""

    def _reduce_func(self, arr):
        return cp.median(arr[-self.N:], 0)


class OrdinalTransformer(BaseTransformer):
    """Encode some of the columns from dataframe as ordinal values."""
    def __init__(self, cols):
        self.cols = cols
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        self.enc.fit(X[self.cols])
        return self

    def transform(self, X, y=None):
        Xord = self.enc.transform(X[self.cols])
        other_cols = [col for col in X.columns if col not in self.cols]
        df = X[other_cols]
        for i, col in enumerate(self.cols):
            df[col] = Xord[:, i]
        return df
