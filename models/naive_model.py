import pandas as pd
import copy
import pickle


class NaiveModel:
    def __init__(self):
        self.data = dict()

    def fit(self, data):
        aux_data = data.mean(axis=0)

        aux_data = aux_data.to_dict()

        self.data = aux_data

    def predict(self, data):
        new_data = copy.deepcopy(data)

        aux_data = new_data.mean(axis=0)

        aux_data = aux_data.to_dict()

        result_data = dict()

        for key in aux_data:
            result_data[key] = aux_data[key] / self.data[key]

        return result_data

    def save(self, path_file_data):
        file_handler = open(path_file_data, 'wb')
        pickle.dump(self.data, file_handler, 0)

    def load(self, path_file_data):
        file_handler = open(path_file_data, 'r')
        self.data = pickle.load(file_handler)


