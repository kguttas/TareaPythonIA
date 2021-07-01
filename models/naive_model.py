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

        for key in new_data:

            divider = 1

            if self.data[key] != 0:
                divider = self.data[key]

            new_data[key] = new_data[key].div(divider)

        return new_data

    def save(self, path_file_data):
        file_handler = open(path_file_data, 'wb')
        pickle.dump(self.data, file_handler, 0)
        file_handler.close()

    def load(self, path_file_data):
        file_handler = open(path_file_data, 'rb')
        self.data = pickle.load(file_handler)
        file_handler.close()



