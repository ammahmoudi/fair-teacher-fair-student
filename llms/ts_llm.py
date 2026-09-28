import abc
from absl import logging
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape


class TimeSeriesLLM(abc.ABC):
    def __init__(
            self,
            name="time_series_llm"
    ):
        self.llm_model = None
        self._name = name

    @abc.abstractmethod
    def predict(
            self,
            test_data
    ):
        pass

    @abc.abstractmethod
    def train(
            self,
            train_data
    ):
        pass

    def evaluate(
            self,
            llm_prediction,
            ground_truth_data,
            metrics=['rmse', 'mae', 'mape']
    ):
        metric_results = {}
        for m in metrics:
            if m == "rmse" or m == "RMSE":
                rmse = calculate_rmse(
                    prediction=llm_prediction,
                    ground_truth=ground_truth_data
                )
                metric_results['rmse'] = rmse
            elif m == "mae" or m == "MAE":
                mae = calculate_mae(
                    prediction=llm_prediction,
                    ground_truth=ground_truth_data
                )
                metric_results['mae'] = mae
            elif m == "mape" or m == "MAPE":
                mape = calculate_mape(
                    prediction=llm_prediction,
                    ground_truth=ground_truth_data
                )
                metric_results['mape'] = mape
            else:
                logging.info("Metric {} not supported.".format(m))
                raise NotImplementedError

        return metric_results

