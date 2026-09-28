import logging
import numpy as np
import pandas as pd
from data_processing.data_scaler import DataScaler
from data_processing.data_sets import Dataset_T1DM, TimeSeriesDataset
from torch.utils.data import DataLoader
from gluonts.dataset.arrow import ArrowWriter

from scripts.data_formatting.core.convert_to_arrow import convert_to_arrow, load_csv_and_prepare_data


class ChronosDataHandler:
    def __init__(
        self,
        settings={
            "input_features": ["feature1", "feature2"],
            "labels": ["label1"],
            "preprocessing_method": "min_max",
            "preprocess_input_features": False,
            "preprocess_label": False,
            "percent": 100,
        },
    ):
        self.scaler = DataScaler(
            settings={
                "input_features": settings["input_features"],
                "labels": settings["labels"],
                "method": settings["preprocessing_method"],
                "preprocess_input_features": settings["preprocess_input_features"],
                "preprocess_label": settings["preprocess_label"],
            }
        )
        self._settings = settings

    def _log_dataset_info(self, dataframe, split, percent):
        """
        Logs the dataset size and split details.
        """
        logging.info(f"Dataset '{split.upper()}' initialized:")
        logging.info(f" - Total samples after applying {percent}%: {len(dataframe)}")
        logging.info(f" - Input features: {self._settings['input_features']}")
        logging.info(f" - Labels: {self._settings['labels']}")

    def load_from_csv(self, path_to_csv, split="train"):
        logging.info(f"Loading data from {path_to_csv} for split '{split}'.")

        dataframe = pd.read_csv(path_to_csv)
        total_rows = len(dataframe)

        # Apply percentage sampling
        percent = self._settings.get("percent", 100)
        if percent < 100:
            rows_to_include = int(total_rows * percent / 100)
            dataframe = dataframe.iloc[:rows_to_include]

        self._log_dataset_info(dataframe, split, percent)

        # Preprocess data if required
        if (
            self._settings["preprocess_input_features"]
            or self._settings["preprocess_label"]
        ):
            if split == "train":
                logging.info("Fitting scaler on train data.")
                self.scaler.fit(dataframe)

            logging.info("Applying scaler to the dataset.")
            dataframe = self.scaler.transform(dataframe)

        return dataframe

    def convert_csv_to_arrow(self, path_to_csv, split="train"):
        logging.info(f"Converting {path_to_csv} to Arrow format for split '{split}'.")

        # Load the data (assuming self.load_from_csv is a method that reads the CSV and returns a DataFrame)
        dataframe = self.load_from_csv(path_to_csv, split=split)

        # # Extract the labels and input features from the settings
        # labels = self._settings["labels"]
        # input_features = self._settings["input_features"]

        # # Ensure there's no overlap between target and input features
        # if set(labels).intersection(input_features):
        #     logging.warning(
        #         f"Target and input features share column names: {set(labels).intersection(input_features)}"
        #     )
        #     # Handle the overlap case here, e.g., separate them or prioritize one column as the target.
        #     # For this example, let's assume we treat the first 'label' as the target and the rest as input features.
        #     input_features = list(set(input_features) - set(labels))

        # # Select relevant columns for input features and target values
        # df_values = dataframe[labels + input_features].values

        # # Ensure the timestamp column is parsed correctly (assuming it's called 'timestamp')
        # timestamps = pd.to_datetime(dataframe["timestamp"],format='%d-%m-%Y %H:%M:%S').values

        # # Convert the DataFrame to a list of series, each with the target values and corresponding timestamps
        # dataset = [
        #     {
        #         "start": timestamps[
        #             i
        #         ],  # Use the first timestamp in each time series as the start time
        #         "target": np.array(value, np.float32),
        #     }
        #     for i, value in enumerate(df_values)
        # ]

        # # Output path for the Arrow file (replace the .csv extension with .arrow)
        arrow_path = path_to_csv.replace(".csv", ".arrow")

        # # Write the dataset to Arrow format with compression
        # ArrowWriter(compression="lz4").write_to_file(dataset, arrow_path)
        
                # Load the data from the CSV file and prepare it
        time_series, timestamps = load_csv_and_prepare_data(path_to_csv)

        # Convert to GluonTS Arrow format and store in a file
        convert_to_arrow(arrow_path, time_series=time_series, timestamps=timestamps)


        logging.info(f"Arrow file saved at {arrow_path}.")
        return arrow_path

    def split_dataframe_input_features_vs_labels(self, dataframe):
        logging.info("Splitting dataframe into input features and labels.")

        input_features = dataframe[self._settings["input_features"]]
        ground_truth = dataframe[self._settings["labels"]]

        logging.info("Split completed:")
        logging.info(f" - Input features shape: {input_features.shape}")
        logging.info(f" - Labels shape: {ground_truth.shape}")

        return input_features, ground_truth


class TimeLLMDataHandler:
    def __init__(
        self,
        settings={
            "input_features": ["feature1", "feature2"],
            "labels": ["label1"],
            "preprocessing_method": "min_max",
            "preprocess_input_features": False,
            "preprocess_label": False,
            "frequency": "5min",
            "num_workers": 1,
            "sequence_length": 6,
            "context_length": 6,
            "prediction_length": 6,
            "percent": 100,
            "val_split": 0,  # Default value for val_split
        },
    ):
        # TODO: Fix using our own data scaler.(currently if enabled it fits the standard scaler on train set)
        # self.scaler = DataScaler(
        #     settings={
        #         'input_features': settings['input_features'],
        #         'labels': settings['labels'],
        #         'method': settings['preprocessing_method'],
        #         'preprocess_input_features': settings['preprocess_input_features'],
        #         'preprocess_label': settings['preprocess_label']
        #     }
        # )

        self._settings = settings
        self.scaler = None
        self._settings = settings

    def load_from_csv(self, path_to_csv, batch_size=32, split="train", missed=None,
                      fair_sampling=False, fair_feature='gender', fair_demographics=None,
                      fair_hypo_threshold=70.0, fair_max_oversample=2.4):
        """
        Load data from CSV and prepare DataLoader.
        Handles the case when validation data is not needed (val_split == 0).

        Args:
            fair_sampling:        If True (training split only), apply hypo-prevalence
                                  oversampling via WeightedRandomSampler to reduce
                                  between-group EO Gap (T1 fair teacher training).
            fair_feature:         Demographic feature to group on (default: 'gender').
            fair_demographics:    Optional demographics dict override.
            fair_hypo_threshold:  Glucose threshold for hypoglycaemia detection (mg/dL).
            fair_max_oversample:  Maximum per-window weight multiplier.
        """
        if split == "test":
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True

        # Handle val_split == 0, return empty val_loader
        if self._settings["val_split"] == 0 and split == "val":
            val_loader = DataLoader(
                [],
                batch_size=batch_size,
                shuffle=False,  # No shuffling since the loader is empty
                num_workers=self._settings["num_workers"],
                drop_last=False,  # Avoid dropping last batch in an empty dataset
            )
            return pd.DataFrame(), val_loader
        # Initialize dataset
        data_set = Dataset_T1DM(
            data_path=path_to_csv,
            flag=split,
            size=(
                self._settings["sequence_length"],
                self._settings["context_length"],
                self._settings["prediction_length"],
            ),
            features="S",
            target=self._settings["labels"][0],
            scale=self._settings["preprocess_input_features"],
            scaler=self.scaler,
            timeenc=0,
            freq=self._settings["frequency"],
            percent=self._settings["percent"],
            val_split=self._settings["val_split"],
            missed=missed,
        )

        sampler = None
        if fair_sampling and split == "train":
            try:
                import logging
                from fairness.utils.sampling import build_group_labels, build_hypo_sampler
                group_labels = build_group_labels(
                    data_path=path_to_csv,
                    seq_len=self._settings["sequence_length"],
                    pred_len=self._settings["prediction_length"],
                    val_split=self._settings["val_split"],
                    percent=self._settings["percent"],
                    feature=fair_feature,
                    demographics=fair_demographics,
                )
                if group_labels is not None and len(group_labels) == len(data_set):
                    sampler, info = build_hypo_sampler(
                        data_set, group_labels,
                        hypo_threshold=fair_hypo_threshold,
                        max_oversample=fair_max_oversample,
                    )
                    logging.info(
                        f"⚖️  Fair teacher sampling active: feature='{fair_feature}', "
                        f"hypo_rates={info['group_hypo_rates']}, "
                        f"max_oversample={info['max_oversample']}x"
                    )
                    shuffle_flag = False  # sampler handles ordering
            except Exception as e:
                import logging
                logging.warning(f"Fair sampling setup failed: {e}. Falling back to standard shuffle.")

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag if sampler is None else False,
            sampler=sampler,
            num_workers=self._settings["num_workers"],
            drop_last=drop_last,
        )

        if split == "train":
            self.scaler = data_set.scaler

        return data_set, data_loader


class EcgTimeLLMDataHandler:
    """MIT-BIH ECG data handler aligned with the existing loader structure.

    This keeps the API style similar to TimeLLMDataHandler while serving the new
    ECG classification path.
    """

    def __init__(
        self,
        settings={
            "dataset_dir": "./data/mit-bih-arrhythmia",
            "window_size": 256,
            "batch_size": 64,
            "num_workers": 0,
            "include_duplicate_202": False,
            "beat_index_csv": None,
            "metadata_csv": None,
            "label_mode": "aami5",
            "sampling_rate_hz": 360.0,
            "rr_clip_seconds": 3.0,
        },
    ):
        self._settings = settings

    def load_dataset(self, split="train"):
        from data_processing.data_sets import Dataset_MITBIH_ECG

        return Dataset_MITBIH_ECG(
            dataset_dir=self._settings["dataset_dir"],
            flag=split,
            window_size=self._settings["window_size"],
            beat_index_csv=self._settings.get("beat_index_csv"),
            metadata_csv=self._settings.get("metadata_csv"),
            include_duplicate_202=self._settings.get("include_duplicate_202", False),
            label_mode=self._settings.get("label_mode", "aami5"),
            sampling_rate_hz=self._settings.get("sampling_rate_hz", 360.0),
            rr_clip_seconds=self._settings.get("rr_clip_seconds", 3.0),
        )

    def load_from_index(
        self,
        split="train",
        batch_size=None,
        shuffle=None,
        fair_sampling=False,
        fair_feature="sex",
        fair_max_oversample=4.0,
        class_balanced_sampling=False,
        class_balanced_max_oversample=50.0,
    ):
        dataset = self.load_dataset(split=split)
        if batch_size is None:
            batch_size = self._settings.get("batch_size", 64)
        if shuffle is None:
            shuffle = split == "train"
        sampler = None
        if fair_sampling and split == "train":
            try:
                from fairness.utils.ecg_sampling import (
                    build_group_class_sampler,
                    build_group_labels_from_samples,
                )

                group_labels = build_group_labels_from_samples(dataset._dataset, fair_feature)
                sampler, info = build_group_class_sampler(
                    dataset._dataset,
                    group_labels,
                    max_oversample=fair_max_oversample,
                )
                logging.info(
                    f"⚖️  ECG fair teacher sampling active: feature='{fair_feature}', "
                    f"max_oversample={info['max_oversample']}x"
                )
                shuffle = False
            except Exception as e:
                logging.warning(f"ECG fair sampling setup failed: {e}. Falling back to standard shuffle.")
        elif class_balanced_sampling and split == "train":
            try:
                from fairness.utils.ecg_sampling import build_class_balanced_sampler

                sampler, info = build_class_balanced_sampler(
                    dataset._dataset,
                    max_oversample=class_balanced_max_oversample,
                )
                logging.info(
                    f"⚖️  ECG class-balanced sampling active: max_oversample={info['max_oversample']}x"
                )
                shuffle = False
            except Exception as e:
                logging.warning(f"ECG class-balanced sampling setup failed: {e}. Falling back to standard shuffle.")
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self._settings.get("num_workers", 0),
            drop_last=False,
        )
        return dataset, data_loader
