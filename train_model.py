import argparse
import pandas as pd
import configparser
from models import naive_model

print("Train Model Script")

path_config_file = None

# Get config argument
parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file og config")

args = parser.parse_args()

path_config_file = args.conf

# Get config variables from file
config = configparser.ConfigParser()

config.read(path_config_file)

# Get data from csv file with training data
data_mnist = pd.read_csv(config['paths']['path_data_train'])

# Instance model
model = naive_model.NaiveModel()

# Training model
model.fit(data_mnist)

# Save model pretraining
model.save(config['paths']['path_save_avg_data_train'])
