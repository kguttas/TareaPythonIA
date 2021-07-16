import argparse
import pandas as pd
import configparser
from models import naive_model


print("Inference Model Script")

path_config_file = None

# Get config argument
parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file config")

parser.add_argument("-datain", type=str, help="path to file for get data")

parser.add_argument("-dataout", type=str, help="path to file for save data")

args = parser.parse_args()

path_config_file = args.conf

path_data_in = args.datain

path_save_predict = args.dataout

# Get config variables from file
config = configparser.ConfigParser()

config.read(path_config_file)

# Instance model
model = naive_model.NaiveModel()

# Load model pretraining
model.load(config['paths']['path_save_avg_data_train'])

# Read data from csv file with testing data
data_mnist = pd.read_csv(path_data_in)

# Do prediction with model pretraining
result_predict = model.predict(data_mnist)

# Sava data predicted in csv file
result_predict.to_csv(path_save_predict)

