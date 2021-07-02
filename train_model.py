import argparse
import pandas as pd
import configparser
from models import naive_model


path_config_file = None

parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file og config")

args = parser.parse_args()

path_config_file = args.conf

config = configparser.ConfigParser()

config.read(path_config_file)

data_mnist = pd.read_csv(config['paths']['path_data_train'])

model = naive_model.NaiveModel()

model.fit(data_mnist)

model.save(config['paths']['path_save_avg_data_train'])
