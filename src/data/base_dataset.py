# src/data/base_dataset.py
"""
This script implements the BaseDataset class.
"""
import src.data.data_generators as dg

class BaseDataset():
    """
    BaseDataset class to represent dataset of financial values.
    """
    def __init__(self, path):
        """
        Initialize BaseDataset class:
        :param path:    str - The path to the data file to use.
        """
        self.dataframe = dg.set_datetime_index(dg.load_data(path))
    
    def get_dataframe(self):
        """
        Returns the dataframe of the BaseDataset instance.
        """
        return self.dataframe

    def head(self):
        """ 
        Returns the head of the DataFrame.
        """
        return self.dataframe.head()

    def info(self):
        """
        Returns info about the DataFrame.
        """
        return self.dataframe.info()

    def returns(self, feature: str, period: str):
        """
        Generates returns of given feature in period.
        :param feature: str - The feature of the dataset to use.
        :param period:  srt - The period to compute returns over.
        """
        return dg.generate_returns(self.dataframe[feature], period)

    def log_price(self, feature):
        """
        Generates Logarithmic prices of given feature.
        :param feature: str - The feature to compute log prices on.
        """
        return dg.generate_log_price(self.dataframe[feature])

    def log_returns(self, feature: str, period: str):
        """
        Generates Logarithmic returns of given feature over period.
        :param feature: str - The feature to compute log returns on.
        :param period:  stc - The period to compute the log returns over.
        """
        return dg.generate_log_returns(self.dataframe[feature], period)

    def realized_variance(self, feature: str, period: str):
        """
        Generates Realized Variance of feature in period.
        :param feature: str - The feature to generate realized variance of.
        :param period:  str - The period to generate realized variance over.
        """
        return dg.generate_realized_var(self.dataframe[feature], period)

    def volume(self, period: str):
        """
        Generates the volume within given period.
        :param period: str - The period to generate volume in.
        """
        return dg.sum_period(self.dataframe['Volume'], period)
        
