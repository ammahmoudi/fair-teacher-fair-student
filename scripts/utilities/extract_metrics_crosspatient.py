import os
import re
import csv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

                    # Use regex to extract seed and model from folder name
                    match = re.search(r"seed_(\d+)_model_([^_]+)", experiment_folder)
                    if not match:
                        raise ValueError(f"Could not parse seed/model from folder name: {experiment_folder}")

                    experiment_params["seed"] = match.group(1)
                    experiment_params["model"] = match.group(2)

                    # Add patient_id from the patient folder
                    # Extract train and test patient IDs from folder name like 'patient_570_train_570_test'
                    # Extract train and test patient IDs from folder name like 'patient_584_train_584_test'
                    match = re.search(r"(\d+)_train_(\d+)_test", patient_folder)
                    if not match:
                        raise ValueError(f"Could not parse train/test IDs from folder name: {patient_folder}")
                    train_patient_id = match.group(1)
                    test_patient_id = match.group(2)

                    experiment_params["train_patient_id"] = train_patient_id
                    experiment_params["test_patient_id"] = test_patient_id


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
                    try:
                        corrected_metrics_str=corrected_metrics_str.replace('inf','0')
                        # corrected_metrics_str.replace('inf','0')
                        corrected_metrics = eval(
                            corrected_metrics_str
                        )  # Convert safely
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
                        str(value)
                        for value in list(experiment_params.values())
                        + [
                            str(corrected_metrics.get("rmse")),
                            str(corrected_metrics.get("mae")),
                            str(corrected_metrics.get("mape")),
                        ]
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
    "./experiment_configs_crosspatient/",
    "experiment_results_crosspatient.csv",
)
