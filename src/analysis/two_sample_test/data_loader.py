import pandas as pd
import numpy as np
import logging

class DataLoader(object):

    def __init__(self) -> None:
        super().__init__()
    
    def loadKPI(self, file_name):
        data = pd.read_csv(file_name)
        for column in data.columns.tolist():
            check_nan_column = data[column].isnull().sum()
            if check_nan_column > 0:
                logging.warn('Find NaN values, using pd.interpolate to replace the NaN values')
                data.interpolate()
                break
        return data
    
    def loadCFG(self, file_name):
        data = pd.read_csv(file_name)
        for column in data.columns.tolist():
            check_nan_column = data[column].isnull().sum()
            if check_nan_column > 0:
                logging.warn('Find NaN values, using pd.interpolate to replace the NaN values')
                data.interpolate()
                break
        return data