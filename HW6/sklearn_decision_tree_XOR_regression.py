from sklearn import tree

X = [[0,0],[0,1],[1,0],[1,1]]
Y = [0,1,1,0]

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,Y)
print(clf)
print("Predicting: {0}".format([[2.,2.]]))
print(clf.predict([[2.,2.]]))
print("Predicting: {0}".format([[1.,-1.]]))
print(clf.predict([[1.,-1.]]))
print("Predicting: {0}".format([[2.,0.]]))
print(clf.predict([[2.,0.]]))