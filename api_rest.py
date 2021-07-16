from flask import Flask, redirect, url_for, request
from models import naive_model
import pandas as pd
import argparse
import configparser


# Get config argument
parser = argparse.ArgumentParser()

parser.add_argument("-conf", type=str, help="path to file og config")

args = parser.parse_args()

# Get config variables from file
config = configparser.ConfigParser()

path_config_file = args.conf

config.read(path_config_file)

print("API REST Naive Model")

app = Flask(__name__)


@app.route('/fit', methods=['POST'])
def fit():
    """This method does training data passed in JSON format"""

    # Get data from request in JSON format
    request_data = request.get_json()

    # Instance model
    model = naive_model.NaiveModel()

    # Convert data in JSON format to DataFrame
    data_mnist = pd.DataFrame.from_dict(request_data, orient="index")

    # Perform training with data passed
    model.fit(data_mnist)

    return data_mnist.to_json()


@app.route('/predict', methods=['POST'])
def predict():
    """This method does prediction from pretraining model"""

    # Get data from request in JSON format
    request_data = request.get_json()

    # Instance model
    model = naive_model.NaiveModel()

    # Load pretraining model
    model.load(config['paths']['path_save_avg_data_train'])

    # Convert data in JSON format to DataFrame
    data_mnist = pd.DataFrame.from_dict(request_data, orient="index")

    # Do prediction with model pretraining
    result_predict = model.predict(data_mnist)

    return result_predict.to_json()


if __name__ == '__main__':
    app.run(debug=True)


