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
import shutil


class Plot(Autoeda):
    def __init__(self,Autoeda):
        self.Autoeda=Autoeda

    def pdfplot(self):
        path = 'C:/Users/LENOVO/Desktop/Automate/venv/'
        DEST_DIR = 'C:/Users/LENOVO/Desktop/plotpdfs/'

        for fname in os.listdir(path):
            if fname.lower().endswith('.pdf'):
                shutil.move(os.path.join(path, fname), DEST_DIR)


        for dirpath, dirname, filenames in walk(DEST_DIR):
            print(filenames)
        merger = PdfFileMerger(DEST_DIR)
        for files in filenames:
            merger.append(DEST_DIR + files)
        merger.write(DEST_DIR + 'out.pdf')
        merger.close()



    def dist(self):
        for col in self.Autoeda.numerical_columns:
            sns.jointplot(x=col, y=self.Autoeda.target, data=self.Autoeda.train, kind="reg")
            plt.figure()
            plt.title("jointplot")
            plt.savefig("distplot.pdf",bbox_inches='tight')
            plt.close()
            plt.show()

    def missing(self):
        missing = self.Autoeda.train.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar()
        plt.savefig("missing.pdf", bbox_inches='tight')
        plt.close()
        plt.show()

    def histogram(self):

        for col in self.Autoeda.numerical_columns:
            condition_pivot = self.Autoeda.train.pivot_table(index=col, values=self.Autoeda.target)
            condition_pivot.plot(kind='hist', color='red', bins=20)
            plt.xlabel(col)
            plt.ylabel(self.Autoeda.target)
            plt.title("Histogram")
            plt.savefig(str(col)+"Histogram.pdf", bbox_inches='tight')
            plt.close()
            plt.show()


    def Scatter(self):
        for col in self.Autoeda.numerical_columns:
            plt.xlabel(col)
            plt.ylabel(self.Autoeda.target)
            plt.title("Scatter")
            plt.scatter(data=self.Autoeda.train, x=col, y=self.Autoeda.target)
           # plt.rcParams['figure.figsize'] = 20, 20
            plt.savefig("scatter.pdf", bbox_inches='tight')
            plt.close()
            plt.show()

    def Boxplot(self):
        for col in self.Autoeda.numerical_columns:
            plt.xlabel(col)
            plt.ylabel(self.Autoeda.target)
            plt.title("Boxplot")
            sns.boxplot(data=self.Autoeda.train, x=col, y=self.Autoeda.target)
            plt.rcParams['figure.figsize'] = 10, 20
            plt.savefig("Boxplot.pdf", bbox_inches='tight')
            plt.close()
            plt.show()

    def spearman(self):
        frame = self.Autoeda.train
        features = self.Autoeda.numerical_columns
        spr = pd.DataFrame()
        spr['feature'] = features
        spr['spearman'] = [frame[f].corr(frame[self.Autoeda.target], 'spearman') for f in features]
        spr = spr.sort_values('spearman')
        plt.figure(figsize=(6, 0.25 * len(features)))
        sns.barplot(data=spr, y='feature', x='spearman', orient='h')
        plt.savefig("spearman.cont.pdf", bbox_inches='tight')
        plt.close()


