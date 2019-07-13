import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
#load the iris dataset 
data=load_iris()
#describe the iris dataset
print("Feature Names  are")
print(data.feature_names)
print("Target Names are")
print(data.target_names)
x,y=data.data,data.target
#splitting data set for training and testing
#here we split dtaa set into 70% for training and 30% for testing
train_x, test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)
score=[]

#Logistic Regression
print("\nLOGISTIC-->")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_x,train_y)
#predict the output by providing test_x data set
p_y=lr.predict(test_x)
#Acuuracy
a=accuracy_score(test_y,p_y)
print("Accuracy: ",a)
score.append(a)


#KNN Algorithm
print("\nKNN-->")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#training the dataset
neigh.fit(train_x,train_y)
#predict the output by providing test_x data set
p_y=neigh.predict(test_x)
#Accuracy
a=accuracy_score(test_y,p_y)
print("Accuracy: ",a)
score.append(a)

#SVM Algorithm
print("\nSVM-->")
from sklearn.svm import SVC
clf = SVC()
#training the dataset
clf.fit(train_x,train_y)
#predict the output by providing test_x data set
p_y=clf.predict(test_x)
#Accuracy
a=accuracy_score(test_y,p_y)
print("Accuracy: ",a)
score.append(a)

#Gaussian Naive
print("\nGUASSIAN NAIVE-->")
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#training the dataset
clf.fit(train_x,train_y)
#predict the output by providing test_x data set
p_y=clf.predict(test_x)
#Accuracy
a=accuracy_score(test_y,p_y)
print("Accuracy: ",a)
score.append(a)

#Decision tree
print("\nDECISION TREE-->")
from sklearn import tree
clf = tree.DecisionTreeClassifier()
#training the dataset
clf.fit(train_x,train_y)
#predict the output by providing test_x data set
p_y=clf.predict(test_x)
#Accuracy
a=accuracy_score(test_y,p_y)
print("Accuracy: ",a)
score.append(a)

algo=['Logistic Regression','KNN','SVM','Guassian Naive','Decision Tree']

#Plotting the graph
plt.plot(algo,score,marker='o')
plt.show()

