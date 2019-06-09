from builtins import *
#from .main import Autoeda
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

class Categorical_plot(Autoeda):
    def __init__(self,Autoeda):

        self.Autoeda=Autoeda

    def percentplot(self):
        for col in self.Autoeda.categorical_columns:
            sns.barplot(x=col, y=self.Autoeda.target, data=self.Autoeda.train, estimator=lambda x: len(x) / len(self.Autoeda.train) * 100)
            plt.xlabel(col)
            plt.ylabel(self.Autoeda.target)
            plt.title(str(col) + 'percentage plot')
            plt.savefig(str(col)+'cat_reg_percentplot.pdf')
            plt.close()
            #plt.show()

    def countplot(self):
        for col in self.Autoeda.categorical_columns:
            sns.countplot(data=self.Autoeda.train, x=col)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title(str(col) + 'count plot')
            plt.savefig(str(col)+'cat_reg_countplot.pdf')
            plt.close()
           # plt.show()

    def uniquevalue(self):
        for col in self.Autoeda.categorical_columns:
            print("Unique", col, self.Autoeda.train[col].nunique())


    def pieplot(self):
        for col in self.Autoeda.categorical_columns:
            per = (self.Autoeda.train[col].value_counts() / self.Autoeda.train[col].count()) * 100

            labels = []
            cent = []
            for fea, cou in per.iteritems():
                labels.append(fea)
                cent.append(cou)

            plt.pie(x=per, labels=labels, autopct='%.0f%%', data=self.Autoeda.train)
            plt.axis("equal")
            plt.title(str(col) + 'pie plot distribution')
            plt.savefig(str(col)+'cat_cla_pie.pdf')
            plt.show()

    def mostFrequent(self):

        col = data[x]
        for col in self.Autoeda.categorical_columns:
            x = self.Autoeda.categorical_columns
            data[col[x]
            print("Most Common", self.Autoeda.categorical_columns,
                  [self.Autoeda.categorical_columns].value_counts().max())

    def validvalue(self):
        for col in self.Autoeda.categorical_columns:
            print("Valid", col, self.Autoeda.train[col].count())


    def nullvalue(self):
        for col in self.Autoeda.categorical_columns:
            print("Missing", col, self.Autoeda.train[col].isnull().sum())