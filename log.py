#import numpy librayry
import numpy as np

#import pandas librayry
import pandas as pd

#import matplotlib librayry
import matplotlib.pyplot as plt

#import seaborn librayry
import seaborn as sns

#incase you get wariningfuture error import warnings Future 
from warnings import simplefilter
simplefilter(action='ignore',category=FutureWarning)

#import sklearn librayry
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#read file
df=pd.read_csv('gre.csv')

x=df[['gmat','gpa','work_experience']]

#y is the prediction column
y=df['admitted']
#show the plot
sns.pairplot(ab,x_vars=['TV','radio','newspaper'],y_vars='sales',height=5,aspect=0.6,kind='reg')
plt.show()
#split the x and y values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#LogisticRegresssion
lr=LogisticRegression()
#fitting values x and y 
lr.fit(x_train,y_train)
#predicting x_test
y_pred=lr.predict(x_test)

print(lr.intercept_)
print(lr.coef_)
print(lr.score(x,y))

#showing the crosstable in y_test and y_pred
mat=pd.crosstab(y_test,y_pred,rownames=['Auctal'],colnames=['predicted'])
sns.heatmap(mat,annot=True)
plt.show()
print()

#print said by said actual and predicted values
df=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
print(df)

#making  the accuray
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))

#making the mean squared error
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#test the prediction
a=660
b=3.3
c=6
abc=[[a,b,c]]
xyz=lr.predict(abc)
print(xyz)






