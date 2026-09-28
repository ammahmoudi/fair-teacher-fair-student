import logging
import pandas as pd
from torch.utils.data import Dataset
from data_processing.data_scaler import StandardScaler
from utils.timefeatures import time_features
import numpy as np
import random
import numpy as np
import random

class Dataset_T1DM(Dataset):
    def __init__(
        self,
        flag="train",
        size=None,
        features="S",
        data_path="train.csv",
        target="_value",
        scale=True,
        timeenc=0,
        freq="5min",
        percent=100,
        seasonal_patterns=None,
        scaler=None,
        val_split=0,
        missed=None
    ):
        """
        Dataset for T1DM glucose levels.

        :param flag: 'train', 'val', or 'test'.
        :param size: Tuple of (sequence_length, context_length, prediction_length).
        :param features: 'S' for single target or 'M' for multiple features.
        :param data_path: Path to the dataset file (train or test file).
        :param target: Target column name (default is '_value').
        :param scale: Whether to scale the data (default is True).
        :param timeenc: Whether to encode time features (0 for no, 1 for yes).
        :param freq: Frequency of the time series (default is '5min').
        :param percent: Percentage of training data (default is 100%).
        :param scaler: external scaler (used to pass the train dataset fitted scaler to the test dataset)
        :param val_split: Fraction of the data (in percentage) to be used as validation set (default is 0 for no validation data).
        """
        if size is None:
            self.sequence_length = (
                12  # Default: 12 samples (60 minutes if every 5 mins)
            )
            self.context_length = 6  # Default: 6 samples (30 minutes)
            self.prediction_length = 12  # Default: Predict 12 samples (60 minutes)
        else:
            self.sequence_length = size[0]
            self.context_length = size[1]
            self.prediction_length = size[2]

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.seasonal_patterns = seasonal_patterns
        self.scaler = scaler
        self.val_split = val_split  # New parameter for validation split
        self._data_transformed = False  # Initialize the flag for scaling status
        self.missed=missed

        self.__read_data__()
        self._log_dataset_info()

        self.enc_in = self.data_x.shape[-1]  # Number of input features
        self.tot_len = (
            len(self.data_x) - self.sequence_length - self.prediction_length + 1
        )  # Total valid sequences

    def _log_dataset_info(self):
        """
        Logs the dataset size and split details.
        """
        mode = self.flag.upper()

        # Calculate the number of training and validation samples
        num_samples = len(self.data_x)
        # num_train = int(num_samples * (100 - self.val_split) / 100)
        # num_val = int(num_samples * self.val_split / 100)

        # Log dataset details
        logging.info(f"Dataset '{mode}' initialized:")
        logging.info(f" - Total samples after applying {self.percent}% and spliting: {num_samples}")
        # if self.flag =='train':
        # logging.info(f" - Training samples: {num_train}")
        # elif self.flag == 'valid':
        # if self.val_split > 0:
            # logging.info(f" - Validation samples: {num_val}")
        
        logging.info(
            f" - Sequence Length: {self.sequence_length}, "
            f"Context Length: {self.context_length}, "
            f"Prediction Length: {self.prediction_length}."
        )
        logging.info(f" - Target Feature: {self.target}")
        logging.info(f" - Scaling Enabled: {self.scale}")
        logging.info(f" - Validation Split: {self.val_split}%")

    def __read_data__(self):
        """
        Reads and preprocesses the data.
        """
        # Load the data
        df_raw = pd.read_csv(self.data_path)
        print(df_raw.head())

        # Ensure correct columns exist
        assert (
            "timestamp" in df_raw.columns and self.target in df_raw.columns
        ), "Dataset must contain 'timestamp' (timestamp) and target columns."

        # Sort by timestamp - handle multiple date formats
        try:
            # Try dd-mm-yyyy format first (training data format)
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], format="%d-%m-%Y %H:%M:%S")
        except ValueError:
            try:
                # Try yyyy-mm-dd format (testing data format)
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], format="%Y-%m-%d %H:%M:%S")
            except ValueError:
                # Fall back to automatic inference
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw = df_raw.sort_values("timestamp")

        # Apply the percent parameter to reduce the dataset size
        total_rows = len(df_raw)
        rows_to_include = int(total_rows * self.percent / 100)
        df_raw = df_raw.iloc[:rows_to_include]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        # Extract values
        data = df_data.values

        # Calculate number of training and validation samples
        num_samples = len(data)
        num_train = int(num_samples * (100 - self.val_split) / 100)
        num_val = int(num_samples * self.val_split / 100)

        if self.flag == "train":
            border1, border2 = 0, num_train
        elif (
            self.flag == "val"
        ):  # Only allow val mode if val_split > 0
            border1, border2 = num_train, num_train + num_val

        else:  # Test mode or if val_split is 0
            border1, border2 = 0, num_samples

        data = data[border1:border2]

        # Use external scaler if provided, else fit during training (if use different x and y fix this to scale properly)
        if self.scale:
            if self.flag == "train" and self.scaler is None:
                # Fit a new scaler during training
                self.scaler = StandardScaler()
                self.scaler.fit(data)
                data = self.scaler.transform(data)
                self._data_transformed = True
            elif self.scaler:  # Use the provided scaler
                data = self.scaler.transform(data)
                self._data_transformed = True
            elif self.flag in ["val", "test"]:
                # Defer scaling if scaler is not yet provided
                self._raw_data = data
                self._data_transformed = False

        # Process time features
        df_stamp = df_raw.iloc[border1:border2][["timestamp"]]
        if self.timeenc == 0:
            # Manually extract time-related features
            df_stamp["month"] = df_stamp["timestamp"].dt.month
            df_stamp["day"] = df_stamp["timestamp"].dt.day
            df_stamp["weekday"] = df_stamp["timestamp"].dt.weekday
            df_stamp["hour"] = df_stamp["timestamp"].dt.hour
            df_stamp["minute"] = df_stamp["timestamp"].dt.minute // (
                60 // 12
            )  # Convert minutes into bins (5-min intervals)
            self.data_stamp = df_stamp[
                ["month", "day", "weekday", "hour", "minute"]
            ].values
        elif self.timeenc == 1:
            # Use a learned encoding for time features
            self.data_stamp = time_features(
                pd.to_datetime(df_stamp["timestamp"].values), freq=self.freq
            )
            self.data_stamp = self.data_stamp.transpose(1, 0)

        if self.missed is not None:
            data = self.apply_missingness(data)
            print('test')
        
        self.data_x = data
        self.data_y = data
        # Calculate number of valid sliding windows
        min_required = self.sequence_length + self.prediction_length
        self.fallback_mode = False  # Add a flag for fallback mode

        if len(data) < min_required:
            if self.flag == "test":
                logging.warning(
                    f"Not enough data for sliding windows. "
                    f"Required: {min_required}, Available: {len(data)}. Using fallback mode for test."
                )
                self.fallback_mode = True
                self.tot_len = 1  # Only one sample: the entire available sequence
            else:
                raise ValueError(
                    f"Not enough data for training/validation. "
                    f"Need at least {min_required}, got {len(data)} rows."
                )
        else:
            self.tot_len = len(data) - self.sequence_length - self.prediction_length + 1

    
    def apply_missingness(self, data : np.ndarray, miss_rate=0.1, missing_type='periodic'):
        if missing_type == 'random':
            num_values = data.size
            num_missing = int(num_values * miss_rate)
            missing_indices = random.sample(range(num_values), num_missing)
            data_flat = data.flatten()
            for idx in missing_indices:
                data_flat[idx] = 0
            data = data_flat.reshape(data.shape)
        elif missing_type == 'periodic':
            window_size = 6
            num_rows = data.shape[0]
            num_missing = int(num_rows * miss_rate)
            num_periods = max(1, num_missing // window_size)
            step = max(1, num_rows // num_periods)
            data_flat = data.flatten()
            for start_idx in range(0, num_rows, step):
                if start_idx + window_size <= num_rows:
                    data_flat[start_idx:start_idx + window_size] = 0
            data = data_flat.reshape(data.shape)
        else:
            raise ValueError("Invalid missing type. Choose either 'random', 'synthetic', or 'periodic'.")
        return data

    def __getitem__(self, index):
        if self.fallback_mode:
            # Use the entire available data as one sequence
            full_seq_x = self.data_x[: -self.prediction_length]
            full_seq_y = self.data_y[-(self.context_length + self.prediction_length):]
            full_seq_x_mark = self.data_stamp[: -self.prediction_length]
            full_seq_y_mark = self.data_stamp[-(self.context_length + self.prediction_length):]
            return full_seq_x, full_seq_y, full_seq_x_mark, full_seq_y_mark

        # Standard mode with sliding windows
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.sequence_length
        r_begin = s_end - self.context_length
        r_end = r_begin + self.context_length + self.prediction_length

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return self.tot_len if not self.fallback_mode else 1


    def inverse_transform(self, data):
        """
        Reverses the scaling transformation for interpretability.

        :param data: Scaled data.
        :return: Original data in the original scale.
        """
        if self.scale:
            return self.scaler.inverse_transform(data)
        else:
            return data

    def set_scaler(self, scaler):
        """
        Set an external scaler for validation or test datasets and transform the data if not already scaled.

        :param scaler: Pre-fitted scaler (e.g., from the training data).
        """
        self.scaler = scaler
        if self.scaler and self.scale:
            # Ensure the data is transformed only if it hasn't already been scaled
            if not hasattr(self, "_data_transformed") or not self._data_transformed:
                self.data_x = self.scaler.transform(self.data_x)
                self.data_y = self.scaler.transform(self.data_y)
                self._data_transformed = True  # Mark that the data has been transformed


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        path_to_csv,
        size,
        features,
        targets,
        flag="train",
        scaler=None,
        timeenc={"timestamp": 0},
        freq="h",
        percent=100,
    ):
        assert (
            len(size) == 3
        ), "Size should be a iterable of 3 integers, e.g.,  [sequence_length, context_length, prediction_length]"
        self.sequence_length = size[0]
        self.context_length = size[1]
        self.prediction_length = size[2]

        assert flag in [
            "train",
            "test",
            "val",
        ], "Flag should be either 'train', 'test' or 'val'"
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.targets = targets
        self.scaler = scaler
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.path_to_csv = path_to_csv
        self.scaler = scaler
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = (
            len(self.data_x) - self.sequence_length - self.prediction_length + 1
        )

    def __read_data__(self):
        df_raw = pd.read_csv(self.path_to_csv)
        # TODO: substract sequence length
        border1s = [0, 0.7 * len(df_raw), 0.9 * len(df_raw)]
        border2s = [0.7 * len(df_raw), 0.9 * len(df_raw), len(df_raw)]

        border1 = int(border1s[self.set_type])
        border2 = int(border2s[self.set_type])

        if self.set_type == 0:
            border2 = (
                border2 - self.sequence_length
            ) * self.percent // 100 + self.sequence_length

        df_data = df_raw[self.features]
        df_data = df_data[border1s[0] : border2s[0]]

        if self.scaler is not None:
            if self.set_type == 0:
                self.scaler.fit(df_data)
            data = self.scaler.transform(df_raw)
        else:
            data = df_data

        df_stamp = df_raw[self.timeenc.keys()][border1:border2]
        df_stamp[self.timeenc.keys()] = pd.to_datetime(df_stamp[self.timeenc.keys()])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.sequence_length
        r_begin = s_end - self.context_length
        r_end = r_begin + self.context_length + self.prediction_length
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (
            len(self.data_x) - self.sequence_length - self.prediction_length + 1
        ) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class GroupLabeledDataset(Dataset):
    """Wraps an existing Dataset to also emit a per-window integer group label.

    The wrapped dataset must have a fixed __len__ and return a tuple from
    __getitem__.  This dataset appends one extra element — the group label —
    at the end of that tuple.

    Args:
        dataset:      Any Dataset whose __getitem__ returns a tuple.
        group_labels: 1-D array-like of integer labels, one per window
                      (length must equal len(dataset)).
    """

    def __init__(self, dataset, group_labels):
        import torch
        self.dataset = dataset
        self.group_labels = torch.tensor(group_labels, dtype=torch.long)
        assert len(self.group_labels) == len(dataset), (
            f"group_labels length {len(self.group_labels)} != dataset length {len(dataset)}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = self.dataset[idx]
        return (*items, self.group_labels[idx])


class Dataset_MITBIH_ECG(Dataset):
    """Compatibility wrapper for MIT-BIH ECG classification datasets.

    This mirrors the role of Dataset_T1DM inside the repository while keeping the
    ECG classification path separate from the BG forecasting logic.
    """

    def __init__(
        self,
        dataset_dir,
        flag="train",
        window_size=256,
        beat_index_csv=None,
        metadata_csv=None,
        include_duplicate_202=False,
        label_mode="aami5",
        sampling_rate_hz=360.0,
        rr_clip_seconds=3.0,
    ):
        from data_processing.ecg.dataset import MitBihBeatDataset

        self._dataset = MitBihBeatDataset(
            dataset_dir=dataset_dir,
            split=flag,
            window_size=window_size,
            beat_index_csv=beat_index_csv,
            metadata_csv=metadata_csv,
            include_duplicate_202=include_duplicate_202,
            label_mode=label_mode,
            sampling_rate_hz=sampling_rate_hz,
            rr_clip_seconds=rr_clip_seconds,
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]