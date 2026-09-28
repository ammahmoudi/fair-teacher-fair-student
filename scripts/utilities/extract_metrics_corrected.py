import os
import re
import csv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import math
import ast


def sanitize_metric(value):
    try:
        # Handle string representations of numpy values like np.float32(13.443349)
        if isinstance(value, str) and "np.float" in value:
            # Extract the numeric value from inside parentheses
            match = re.search(r"\((.*?)\)", value)
            if match:
                value = match.group(1)

        val = float(value)
        if math.isinf(val) or math.isnan(val):
            return "NaN"
        return str(val)
    except (ValueError, TypeError):
        return "NaN"


def extract_corrected_metrics_to_csv(base_dir, output_csv):
    logging.info("Starting corrected metrics extraction.")

    # Regex pattern to extract both original and corrected metrics
    metrics_pattern = re.compile(
        r"Original Metrics: (\{.*?\}), Corrected Metrics: (\{.*?\})"
    )
    log_datetime_pattern = re.compile(r"logs_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")

    # List to store extracted results
    results = []
    csv_header = []  # Header will be dynamically created
    existing_rows = set()

    # Check if the file already exists and load existing rows (excluding the header)
    file_exists = os.path.exists(output_csv)
    if file_exists and os.stat(output_csv).st_size > 0:
        with open(output_csv, "r", newline="") as f:
            reader = csv.reader(f)
            csv_header = next(reader, None)  # Read the header
            for row in reader:
                existing_rows.add(tuple(row))  # Store existing rows
        logging.info(f"Loaded {len(existing_rows)} existing rows from {output_csv}.")

    # Walk through all experiment folders
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "log.log":  # Targeting the log files
                log_path = os.path.join(root, file)
                logging.info(f"Processing log file: {log_path}")

                # Extract log date-time from the log folder path
                log_datetime_match = log_datetime_pattern.search(log_path)
                log_datetime = (
                    log_datetime_match.group(1) if log_datetime_match else "Unknown"
                )

                # Extract experiment details from the folder structure
                path_parts = log_path.split(os.sep)

                try:
                    # Extract the experiment folder and patient folder
                    experiment_folder = None
                    patient_folder = None

                    # Identify the experiment folder and patient folder
                    for part in path_parts:
                        if part.startswith("seed_"):
                            experiment_folder = part
                        elif part.startswith("patient_"):
                            patient_folder = part

                    if not experiment_folder or not patient_folder:
                        raise ValueError(
                            "Experiment or patient folder not found in path"
                        )

                    # Parse experiment folder details dynamically
                    experiment_details = experiment_folder.split("_")

                    # Extract the basic experiment parameters from folder structure
                    experiment_params = {}
                    for i in range(0, len(experiment_details), 2):
                        if i + 1 < len(experiment_details):
                            key = experiment_details[i]
                            value = experiment_details[i + 1]
                            experiment_params[key] = value

                    # Add patient_id from the patient folder
                    patient_id = patient_folder.split("_")[1]
                    experiment_params["patient_id"] = patient_id
                    experiment_params["log_datetime"] = (
                        log_datetime  # Add the extracted timestamp
                    )

                    # Read the log file and extract the last metrics line
                    with open(log_path, "r") as f:
                        lines = f.readlines()

                    # Search for the last metrics line
                    corrected_metrics_str = None
                    for line in reversed(lines):
                        match = metrics_pattern.search(line)
                        if match:
                            corrected_metrics_str = match.group(
                                2
                            )  # Extract only corrected metrics
                            break

                    if not corrected_metrics_str:
                        logging.warning(
                            f"No corrected metric results found in {log_path}"
                        )
                        continue  # Skip this file if no metrics were found

                    # Convert extracted corrected metrics to a dictionary
                   
                        # corrected_metrics_str=corrected_metrics_str.replace('inf','0')
                        # corrected_metrics_str.replace('inf','0')
                    import ast

                    sanitized_str = (
                        corrected_metrics_str.replace("inf", '"inf"')
                                            .replace("-inf", '"-inf"')
                                            .replace("nan", '"nan"')
                    )
                    sanitized_str = re.sub(
                            r"np\.float\d*\(([\d\.]+)\)", r'"\1"', sanitized_str
                        )
                    try:
                        corrected_metrics = ast.literal_eval(sanitized_str)
                    except Exception as e:
                        logging.error(f"Error parsing metrics in {log_path}: {e}")
                        continue

                    # Dynamically generate the header if CSV is empty
                    if not csv_header:
                        csv_header = list(experiment_params.keys()) + [
                            "rmse",
                            "mae",
                            "mape",
                        ]

                    # Store the extracted values in the results list
                    result_row = tuple(
                        str(value) for value in list(experiment_params.values())
                    ) + tuple(
                        sanitize_metric(corrected_metrics.get(k))
                        for k in ["rmse", "mae", "mape"]
                    )

                    if result_row not in existing_rows:  # Avoid duplicate rows
                        results.append(result_row)
                        existing_rows.add(result_row)
                        logging.info(f"New data added: {result_row}")
                    else:
                        logging.info("Duplicate entry found, skipping.")

                except Exception as e:
                    logging.error(f"Error processing {log_path}: {e}")

    # Save results to CSV (Append Mode)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if (
            not file_exists or os.stat(output_csv).st_size == 0
        ):  # Write header only if the file is new
            writer.writerow(csv_header)
            logging.info(f"CSV header written: {csv_header}")

        if results:  # Only write if there are new results
            writer.writerows(results)  # Append new rows only if they are not duplicates
            logging.info(f"Appended {len(results)} new rows to {output_csv}")
        else:
            logging.info("No new data to append.")

    logging.info(f"Results saved to {output_csv}")


# Example usage:
# extract_corrected_metrics_to_csv("./experiment_configs_time_llm_inference/", "corrected_metrics_results.csv")

# Example usage:
extract_corrected_metrics_to_csv(
    "./experiment_configs_chronos_training_noised/",
    "experiment_configs_chronos_training_noised_fixed.csv",
)
# extract_corrected_metrics_to_csv(
#     "./experiment_d1namo_configs_chronos_inference/",
#     "experiment_d1namo_results_chronos_inference_fixed.csv",
# )
