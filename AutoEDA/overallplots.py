import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from builtins import *
#from Plot import *
from main import Autoeda
from Impute import *
from matplotlib.pyplot import *
import scipy.stats as st
from scipy import stats
from os import walk
from PyPDF2 import PdfFileMerger,PdfFileReader
import os
import PyPDF2
from matplotlib.backends.backend_pdf import PdfPages

class Overall_plot(Autoeda):
    def __init__(self,Autoeda):
        self.Autoeda=Autoeda

    def pdfplot(self):
        path = 'C:/Users/LENOVO/Desktop/Automate/venv/Plot.py'
        for dirpath, dirname, filenames in walk(path):
            print(filenames)

        merger = PdfFileMerger(path)

        for files in filenames:
            merger.append(path + files)
            merger.write(path + 'out.pdf')
            merger.close()

    def corr(self):
        numerical_columns = self.Autoeda.train.select_dtypes(include=[np.number])
        categorical_columns = self.Autoeda.train.select_dtypes(exclude=[np.number])

        print("The dataset has {0} numerical data and {1} categrical data".format(numerical_columns.shape[1],
                                                                                          categorical_columns.shape[1]))
        #df = pd.DataFrame({'numcols': self.Autoeda.numerical_columns})
        corr = numerical_columns.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr)
        plt.savefig("overall_correlation.pdf",bbox_inches="tight")
        plt.close()
        plt.show()

        # Distribution of target variable.
    def distri(self):
        data = self.Autoeda.train
        x = self.Autoeda.target
        y=data[x]
        # Organize Data.
        SR_y = pd.Series(y)
# Plot the distribution of Target Variable.
        fig, ax = plt.subplots()
        sns.set()
        sns.distplot(SR_y, bins=25, color="g", ax=ax)
        plt.title("Distribution of Target Variable")
        plt.xlabel("target")
        plt.savefig("distribution of target.pdf", bbox_inches='tight')
        plt.close()
        plt.show()

    def spearman(self):
        frame = self.Autoeda.train
        features = self.Autoeda.train.columns
        spr = pd.DataFrame()
        spr['feature'] = features
        spr['spearman'] = [frame[f].corr(frame[self.Autoeda.target], 'spearman') for f in features]
        spr = spr.sort_values('spearman')
        plt.figure(figsize=(6, 0.25 * len(features)))
        sns.barplot(data=spr, y='feature', x='spearman', orient='h')
        plt.savefig("spearman.pdf",bbox_inches='tight')
        plt.close()

    def missval(self):
        total = self.Autoeda.train.isnull().sum().sort_values(ascending=False)
        percent = (self.Autoeda.train.isnull().sum() / self.Autoeda.train.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print (missing_data)


