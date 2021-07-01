import argparse
import pandas as pd

path_config_file = None

parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file og config")

args = parser.parse_args()

path_config_file = args.conf

if path_config_file is not None:
    print(path_config_file)

    import configparser

    config = configparser.ConfigParser()

    config.read(path_config_file)

    data_mnist = pd.read_csv(config['paths']['path_mnist_data'])

    from models import naive_model

    model = naive_model.NaiveModel()

    model.fit(data_mnist)

    model.save(config['paths']['path_save_avg_data_train'])
