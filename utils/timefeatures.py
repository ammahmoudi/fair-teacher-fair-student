from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


import numpy as np
import pandas as pd

def decode_time_features(encoded_features, freq='5min'):
    """
    Decodes normalized time features back into readable timestamps.

    Parameters:
    - encoded_features: np.ndarray
        Encoded time features as a 2D array (shape: [num_features, num_timestamps]).
    - freq: str
        The frequency of the data ('5min', '1H', etc.).

    Returns:
    - pd.DatetimeIndex
        Decoded timestamps corresponding to the encoded features.
    """

    # Convert normalized features back into absolute values
    num_timestamps = encoded_features.shape[1]

    # Initialize a time index (relative for demonstration)
    base_time = pd.Timestamp('1970-01-01 00:00:00')
    time_index = []

    # Extract relevant time components from encoded features
    if freq in ['S', 'T', 'H', '5min']:
        # Hour of day
        if encoded_features.shape[0] >= 1:
            hours = ((encoded_features[0] + 0.5) * 23).astype(int)
        else:
            hours = np.zeros(num_timestamps, dtype=int)

        # Minute of hour
        if encoded_features.shape[0] >= 2:
            minutes = ((encoded_features[1] + 0.5) * 59).astype(int)
        else:
            minutes = np.zeros(num_timestamps, dtype=int)

        # Day of the month
        if encoded_features.shape[0] >= 3:
            days = ((encoded_features[2] + 0.5) * 30 + 1).astype(int)
        else:
            days = np.ones(num_timestamps, dtype=int)

        # Month of the year
        if encoded_features.shape[0] >= 4:
            months = ((encoded_features[3] + 0.5) * 11 + 1).astype(int)
        else:
            months = np.ones(num_timestamps, dtype=int)

        # Year (if needed, assume current year)
        if encoded_features.shape[0] >= 5:
            years = 2022 * np.ones(num_timestamps, dtype=int)
        else:
            years = np.ones(num_timestamps, dtype=int)

        # Reconstruct timestamps
        for i in range(num_timestamps):
            try:
                timestamp = pd.Timestamp(year=years[i], month=months[i], day=days[i],
                                          hour=hours[i], minute=minutes[i])
                time_index.append(timestamp)
            except ValueError:
                time_index.append(base_time)  # Use a fallback for invalid dates

    return pd.DatetimeIndex(time_index)

import pandas as pd

def decode_manual_time_features(encoded_features, year=2022, freq='5min'):
    """
    Decodes manually extracted time features (when timeenc == 0) back to timestamps.

    Parameters:
    - encoded_features: np.ndarray
        Manually extracted time features as a 2D array (shape: [num_timestamps, num_features]).
        Expects columns like [month, day, weekday, hour, minute_bin, ...].
    - year: int
        The reference year for reconstruction.
    - freq: str
        Frequency of the data (e.g., '5min').

    Returns:
    - pd.DatetimeIndex
        Decoded timestamps corresponding to the extracted features.
    """
    decoded_timestamps = []

    # Determine the minute interval from the frequency
    minutes_per_bin = int(pd.Timedelta(freq).seconds / 60)

    for row in encoded_features:  # Iterate over each time step (1D array)
        try:
            # Extract relevant features dynamically
            month = int(row[0])  # Month
            day = int(row[1])    # Day
            hour = int(row[3])   # Hour
            minute_bin = int(row[4])  # Minute bin
            
            # Decode minutes
            minute = minute_bin * minutes_per_bin
            
            # Create the timestamp
            timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
            decoded_timestamps.append(timestamp)
        except Exception as e:
            print(f"Error decoding row {row}: {e}")
            decoded_timestamps.append(None)  # Append None for invalid rows

    return pd.DatetimeIndex(decoded_timestamps)
