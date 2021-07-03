import pandas as pd
import copy
import pickle


class NaiveModel:
    """
    This class is a dummy machine learning model
    """
    def __init__(self):
        """
        Constructor of class
        """
        self.data = dict()

    def fit(self, data_in):
        """
        This method is for training data model
        :param data_in: argument of type DataFrame
        :return:
        """

        # Get mean of columns in dataframe
        aux_data = data_in.mean(axis=0)

        # Convert result to list
        aux_data = aux_data.to_dict()

        # Save in class data
        self.data = aux_data

    def predict(self, data_in):
        """
        This method is for evaluate model
        :param data_in: argument of type DataFrame, this contains new data for do predictions.
        :return:
        """

        # Deep copy data for don't modify input data
        new_data = copy.deepcopy(data_in)

        # Iterate over input data
        for key in new_data:

            divider = 1

            try:
                # Get divider , if this is not equal to zero then use for division
                if self.data[key] != 0:
                    divider = self.data[key]
            except KeyError as e:
                # If index key don't exists then print message of error
                print(f"Index not found, message: {str(e)}")

            # Update column with new data divided for divider
            new_data[key] = new_data[key].div(divider)

        return new_data

    def save(self, path_file_data):
        """
        This method is for save model in file.
        :param path_file_data: path to file
        :return:
        """
        file_handler = open(path_file_data, 'wb')
        pickle.dump(self.data, file_handler, 0)
        file_handler.close()

    def load(self, path_file_data):
        """
        this method is for load model from file
        :param path_file_data: path to file
        :return:
        """
        file_handler = open(path_file_data, 'rb')
        self.data = pickle.load(file_handler)
        file_handler.close()



