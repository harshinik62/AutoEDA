from main import Autoeda
from Plot import Plot
from categorical_plots import Categorical_plot
from overallplots import Overall_plot


easylearn = Autoeda(r'C:\Users\LENOVO\Desktop\titanic data\train.csv',
                    r"C:\Users\LENOVO\Desktop\titanic data\test.csv",
                    target ='Survived',uid = 'PassengerId',
                    column_descriptions = { "Survived" : 'output' })
#easylearn.info()
#easylearn.shape()
easylearn.categorize()
#print(easylearn.numerical_columns)
#easylearn.train.head()
#easylearn.Plot.histogram()


# cont. plots
ply=Plot(easylearn)
#ply.spearman()
#ply.dist()
#ply.missing()
#ply.histogram()
#ply.Scatter()
#ply.Boxplot()
ply.pdfplot()


#cat.plots
cat=Categorical_plot(easylearn)
cat.percentplot()
#cat.countplot()
#cat.mostFrequent()
#cat.uniquevalue()
#cat.validvalue()
#cat.nullvalue()

#cat.pieplot()


#overall plots
over=Overall_plot(easylearn)
#over.missval()
#over.spearman()
#over.corr()
#over.distri()
