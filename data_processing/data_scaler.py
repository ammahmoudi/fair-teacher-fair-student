from absl import logging


class DataScaler:

    def __init__(
        self,
        settings={
            "input_features": ["feature1", "feature2"],
            "labels": ["label1"],
            "method": "min_max",
            "preprocess_input_features": True,
            "preprocess_label": False,
        },
    ):
        self._settings = settings
        self._scaler_parameters = {}

    def fit(self, dataframe):
        if self._settings["method"] == "min_max":
            if self._settings["preprocess_input_features"]:
                for feature in self._settings["input_features"]:
                    self._scaler_parameters[feature] = {
                        "min": dataframe[feature].min(),
                        "max": dataframe[feature].max(),
                    }
            if self._settings["preprocess_label"]:
                for label in self._settings["labels"]:
                    self._scaler_parameters[label] = {
                        "min": dataframe[label].min(),
                        "max": dataframe[label].max(),
                    }
        elif self._settings["method"] == "z_score":
            if self._settings["preprocess_input_features"]:
                for feature in self._settings["input_features"]:
                    self._scaler_parameters[feature] = {
                        "mean": dataframe[feature].mean(),
                        "std": dataframe[feature].std(),
                    }
            if self._settings["preprocess_label"]:
                for label in self._settings["labels"]:
                    self._scaler_parameters[label] = {
                        "mean": dataframe[label].mean(),
                        "std": dataframe[label].std(),
                    }
        else:
            logging.info("Method {} not supported.".format(self._settings["method"]))
            raise NotImplementedError

    def min_max_scale(self, dataframe):
        if self._settings["preprocess_input_features"]:
            for feature in self._settings["input_features"]:
                dataframe[feature] = (
                    dataframe[feature] - self._scaler_parameters[feature]["min"]
                ) / (
                    self._scaler_parameters[feature]["max"]
                    - self._scaler_parameters[feature]["min"]
                )
        if self._settings["preprocess_label"]:
            for label in self._settings["labels"]:
                dataframe[label] = (
                    dataframe[label] - self._scaler_parameters[label]["min"]
                ) / (
                    self._scaler_parameters[label]["max"]
                    - self._scaler_parameters[label]["min"]
                )

        return dataframe

    def min_max_rescale(self, dataframe):
        if self._settings["preprocess_input_features"]:
            for feature in self._settings["input_features"]:
                dataframe[feature] = (
                    dataframe[feature]
                    * (
                        self._scaler_parameters[feature]["max"]
                        - self._scaler_parameters[feature]["min"]
                    )
                    + self._scaler_parameters[feature]["min"]
                )
        if self._settings["preprocess_label"]:
            for label in self._settings["labels"]:
                dataframe[label] = (
                    dataframe[label]
                    * (
                        self._scaler_parameters[label]["max"]
                        - self._scaler_parameters[label]["min"]
                    )
                    + self._scaler_parameters[label]["min"]
                )

        return dataframe

    def z_score_scale(self, dataframe):
        if self._settings["preprocess_input_features"]:
            for feature in self._settings["input_features"]:
                dataframe[feature] = (
                    dataframe[feature] - self._scaler_parameters[feature]["mean"]
                ) / self._scaler_parameters[feature]["std"]
        if self._settings["preprocess_label"]:
            for label in self._settings["labels"]:
                dataframe[label] = (
                    dataframe[label] - self._scaler_parameters[label]["mean"]
                ) / self._scaler_parameters[label]["std"]

        return dataframe

    def z_score_rescale(self, dataframe):
        if self._settings["preprocess_input_features"]:
            for feature in self._settings["input_features"]:
                dataframe[feature] = (
                    dataframe[feature] * self._scaler_parameters[feature]["std"]
                    + self._scaler_parameters[feature]["mean"]
                )
        if self._settings["preprocess_label"]:
            for label in self._settings["labels"]:
                dataframe[label] = (
                    dataframe[label] * self._scaler_parameters[label]["std"]
                    + self._scaler_parameters[label]["mean"]
                )

        return dataframe

    def transform(self, dataframe):
        if self._settings["method"] == "min_max":
            return self.min_max_scale(dataframe)
        elif self._settings["method"] == "z_score":
            return self.z_score_scale(dataframe)

    def inverse_transform(self, dataframe):
        if self._settings["method"] == "min_max":
            return self.min_max_rescale(dataframe)
        elif self._settings["method"] == "z_score":
            return self.z_score_rescale(dataframe)
        else:
            logging.info("Method {} not supported.".format(self._settings["method"]))
            raise NotImplementedError

        return dataframe


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
