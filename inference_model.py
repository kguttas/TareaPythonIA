import argparse
import pandas as pd
import configparser
from models import naive_model


print("Inference Model Script")

path_config_file = None

parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file config")

parser.add_argument("-datain", type=str, help="path to file for get data")

parser.add_argument("-dataout", type=str, help="path to file for save data")

args = parser.parse_args()

path_config_file = args.conf

path_data_in = args.datain

path_save_predict = args.dataout

config = configparser.ConfigParser()

config.read(path_config_file)

model = naive_model.NaiveModel()

model.load(config['paths']['path_save_avg_data_train'])

data_mnist = pd.read_csv(path_data_in)

result_predict = model.predict(data_mnist)

result_predict.to_csv(path_save_predict)

