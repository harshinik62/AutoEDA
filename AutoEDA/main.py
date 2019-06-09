
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
from builtins import *
#from Plot import Plot

from Impute import Imputer
import numpy as np
import pandas as pd

#train = pd.read_csv(r'C:\Users\LENOVO\Desktop\train.csv')

class Autoeda():
    def __init__(self, train, test, target, uid=None,column_descriptions=None, dtype=np.float32):
        train = pd.read_csv(train)
        test = pd.read_csv(test)
        self.train = train
        self.test = test
        self.target = target

        self.uid=uid

       # print(test.head())
        self.dtype = dtype

        if column_descriptions == None:
            column_descriptions = {}

        self.column_descriptions = column_descriptions
        self.drop = set(['ignore', 'output', 'target'])
        self.numerical_columns =[]
        self.num_numerical_cols = None
        self.categorical_columns = []
        self.newcols=[]
        self.numeric_col_types = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#        uid = test.pop(uid)
    #    train =train.drop([uid], axis=1)

   #     self.plot = Plot()

        self.Impute = Imputer()
        self.np = np
        self.pd = pd
    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default
    def data_n():
        """
        Updates train_n and test_n numeric datasets (used for model data creation) based on numeric datatypes from train and test datasets.
        """
        train_n = train.select_dtypes(include=[np.number])
        test_n = test.select_dtypes(include=[np.number])

    def configure(self, option=None, value=None):
        """
        Configure  defaults with ``option`` configuration parameter, ``value`` setting. When method is called without parameters it simply returns the current config dictionary, otherwise returns the updated configuration.
        """
        if option and value:
            _config[option] = value
        return _config

    def EDAinfo(self):
        self.train.info()
#        builtins.print('-' * 40)
        self.test.info()

        data_n()
        eda_metrics = []
        nulls_by_features = train.isnull().sum() + test.isnull().sum()
        nulls = nulls_by_features[1].sum()
        if nulls:
            eda_index.append('Nulls')
            eda_metrics.append([nulls, 'Use feature.impute.'])

        skew = train_n.skew()
        skew_upper = skew[skew > data_n._config['outlier_threshold']]
        skew_lower = skew[skew < data_n._config['outlier_threshold']]
        if not skew_upper.empty:
            eda_index.append('Outliers Upper')
            eda_metrics.append(
                [skew_upper.axes[0].tolist(),
                 'Positive skew (> {}). Use feature.outliers(upper).'.format(
                     data_n._config['outlier_threshold'])])
        if not skew_lower.empty:
            eda_index.append('Outliers Lower')
            eda_metrics.append(
                [skew_lower.axes[0].tolist(),
                 'Negative skew (< -{}). Use feature.outliers(lower).'.format(
                     data_n._config['outlier_threshold'])])

        eda_index.append('Shape')
        feature_by_sample = data_n.train.shape[1] / data_n.train.shape[1]
        message = '#Features / #Samples > {}. Over-fitting.'.format(data_n._config['overfit_threshold'])
        message = message if feature_by_sample < data_n._config['overfit_threshold'] else ''
        eda_metrics.append([self.shape(), message])

        numerical_ratio = builtins.int(data_n.train_n.shape[1] / data_n.train.shape[1] * 100)
        if numerical_ratio < 100:
            eda_index.append('Numerical Ratio')
            eda_metrics.append(['{}%'.format(numerical_ratio),
                                'Aim for 100% numerical.'])
        print (eda_metrics)
    def shape(self):
        """
        Print shape (samples, features) of train, test datasets and number of numerical features in each dataset.
        """

        message = 'train {} | test {}'
        return message.format(self.train.shape, self.test.shape)

    def categorize(self):
        #numerical_columns = []
      # categorical_columns = []
        numeric_col_types = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in self.train.columns:
            col_desc = self.column_descriptions.get(col, False)
            numerical_data = self.train.select_dtypes(include=[np.number])
            if col_desc in ['continuous', 'int', 'float', 'numerical']:
                self.numerical_columns.append(col)
            elif col_desc in self.drop:
               print(col + " ->  " + "this are targets or has to be dropped\n")
                # continue
            elif col_desc == ['category']:
                self.categorical_columns.append(col)
            elif col_desc == False and col in numerical_data:
                col_uni_val = self.train[col].nunique()
                col_tot_val = self.train[col].count()
                likely_cat = col_uni_val / col_tot_val
                if likely_cat < 0.05:
                    self.categorical_columns.append(col)
                # print("cate numeric")
                # print(categorical_columns)
                # print("\n")
                else:
                    self.numerical_columns.append(col)
                # print("numeri")
                # print(numerical_columns)
                # print("\n")
            else:
                self.categorical_columns.append(col)
            # print("non numeric should come here")
            # print(categorical_columns)
            # print("\n")
        print("This are numerical or cont. columns")
        print(self.numerical_columns)
        print("\n")
        print("This are categorical  columns")
        print(self.categorical_columns)
        print("\n")
        self.newcols=self.categorical_columns+self.numerical_columns
        #return self.numerical_columns, self.categorical_columns

