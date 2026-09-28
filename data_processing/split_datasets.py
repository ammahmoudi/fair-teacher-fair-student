import argparse
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_path", dest="csv_path", default=".")
argparser.add_argument("--save_path", dest="save_path", default=".")
argparser.add_argument(
    "--split_probs",
    dest="split_probs",
    nargs='+',
    type=float,
    default=[0.7, 0.3, 0.0]
)
args = argparser.parse_args()

csv_path = args.csv_path
save_path = args.save_path
split_probs = args.split_probs

def split_dataset(
        parameters={
            'data_path': '.',
            'save_path': '.',
            'split_probs': 6,
        }
):
    # load data
    data = pd.read_csv(parameters['data_path'])
    split_data = {}
    for ds_name, split_prob in zip(['train', 'test', 'validation'], parameters['split_probs']):
        if split_prob > 0.0:
            split_data[ds_name] = data.sample(frac=split_prob)
            data = data.drop(split_data[ds_name].index)

    for ds_name, ds in split_data.items():
        ds.to_csv(
            parameters['save_path'] + '/{}.csv'.format(ds_name),
            index=False
        )

if __name__ == '__main__':
    split_dataset(
        parameters={
            'data_path': csv_path,
            'save_path': save_path,
            'split_probs': split_probs
        }
    )