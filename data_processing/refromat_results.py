import os
import pandas as pd


def reformat_results(input_csv_path, output_csv_path):
    """
    Reformats the saved results CSV into a structured format with dynamic `t_i` columns for inputs,
    predictions, and ground truth values. Timestamps are optional.

    Parameters:
    - input_csv_path: str
        Path to the input CSV file with grouped x_timestamps, y_timestamps, inputs, ground truth, and predictions.
    - output_csv_path: str
        Path to save the reformatted CSV file.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv_path)

    # Initialize an empty list to hold reformatted rows
    reformatted_data = []

    for idx, row in df.iterrows():
        # Extract inputs, ground_truth, and predictions
        inputs = list(map(float, row['inputs'].split(', ')))
        ground_truth = list(map(float, row['ground_truth'].split(', ')))
        predictions = list(map(float, row['predictions'].split(', ')))

        # Calculate context and prediction lengths dynamically
        context_length = len(inputs)
        prediction_length = len(ground_truth)

        # Create a row dictionary for this sequence
        reformatted_row = {}

        # # Add x_timestamps and inputs
        # if 'x_timestamps' in df.columns and not pd.isna(row['x_timestamps']):
        #     x_timestamps = row['x_timestamps'].split(', ')
        #     assert len(x_timestamps) == context_length, (
        #         f"Mismatch in x_timestamps and inputs in row {idx}: "
        #         f"x_timestamps({len(x_timestamps)}), inputs({context_length})."
        #     )
        #     for i in range(context_length):
        #         reformatted_row[f't_{i}_timestamp'] = x_timestamps[i]

        # for i in range(context_length):
        #     reformatted_row[f't_{i}_input'] = inputs[i]

        # # Add y_timestamps, ground_truth, and predictions
        # if 'y_timestamps' in df.columns and not pd.isna(row['y_timestamps']):
        #     y_timestamps = row['y_timestamps'].split(', ')
        #     assert len(y_timestamps) == prediction_length, (
        #         f"Mismatch in y_timestamps and predictions/ground_truth in row {idx}: "
        #         f"y_timestamps({len(y_timestamps)}), ground_truth({prediction_length})."
        #     )
        #     for i in range(prediction_length):
        #         pred_idx = context_length + i
        #         reformatted_row[f't_{pred_idx}_timestamp'] = y_timestamps[i]

        for i in range(prediction_length):
            pred_idx = context_length + i
            reformatted_row[f't_{pred_idx}_true'] = ground_truth[i]
            reformatted_row[f't_{pred_idx}_pred'] = predictions[i]

        # Append the reformatted row to the data
        reformatted_data.append(reformatted_row)

    # Convert the reformatted data to a DataFrame
    reformatted_df = pd.DataFrame(reformatted_data)

    # Save the reformatted DataFrame to a new CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    reformatted_df.to_csv(output_csv_path, index=False)
    print(f"Reformatted results saved to {output_csv_path}")
    return reformatted_df
