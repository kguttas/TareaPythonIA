import argparse
import configparser
from sklearn.model_selection import train_test_split
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file config")

args = parser.parse_args()

path_config_file = args.conf

config = configparser.ConfigParser()

config.read(path_config_file)

mnist_data = pd.read_csv(config['paths']['path_mnist_data'])

X_train, X_test = train_test_split(mnist_data, test_size=0.2, random_state=4)

X_train.to_csv(config['paths']['path_data_train'])

X_test.to_csv(config['paths']['path_data_test'])

with open(config['paths']['path_json_train'], 'w') as text_file:
    text_file.write(X_train.to_json(orient='index'))

with open(config['paths']['path_json_test'], 'w') as text_file:
    text_file.write(X_test.to_json(orient='index'))




