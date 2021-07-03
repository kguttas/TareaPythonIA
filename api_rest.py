from flask import Flask, redirect, url_for, request
from models import naive_model
import pandas as pd
import argparse
import configparser

parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file og config")

args = parser.parse_args()

config = configparser.ConfigParser()

path_config_file = args.conf

config.read(path_config_file)

print("API REST Naive Model")

app = Flask(__name__)


@app.route('/fit', methods=['POST'])
def fit():
    request_data = request.get_json()

    model = naive_model.NaiveModel()

    data_mnist = pd.DataFrame.from_dict(request_data, orient="index")

    model.fit(data_mnist)

    return data_mnist.to_json()


@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()

    model = naive_model.NaiveModel()

    model.load(config['paths']['path_save_avg_data_train'])

    data_mnist = pd.DataFrame.from_dict(request_data, orient="index")

    result_predict = model.predict(data_mnist)

    return result_predict.to_json()


if __name__ == '__main__':
    app.run(debug=True)


