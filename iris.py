from sklearn import datasets
from sklearn import tree
import numpy as np

#Taking data from the iris dataset
iris = datasets.load_iris()
clf = tree.DecisionTreeClassifier()

#Training the dataset
clf= clf.fit(iris.data,iris.target)

#Taking data from the user
t = input ("enter the Sepal length, Sepal width, Petal length, Petal width:").split(",")
t = np.array(t,dtype='float')

#Predicting the species name
p = clf.predict(t.reshape(1,-1))
print ("Species:Iris "iris.target_names[p])
