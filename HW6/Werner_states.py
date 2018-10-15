import numpy as np
import random
from sklearn import tree


# generate the easier, more normal dataset
x = []
y = []
for i in range(101):
    if i % 2 == 0:
        p = random.uniform(0.3, 0.9) + random.uniform(-0.1, 0.1)
        x.append([1-p, 0, 0, 0, 0, p+1, -2*p, 0, 0, -2*p, p+1, 0, 0, 0, 0, 1-p])
        y.append(1)
    else:
        p = random.uniform(0.1, 0.3) + random.uniform(-0.1, 0.1)
        x.append([1-p, 0, 0, 0, 0, p+1, -2*p, 0, 0, -2*p, p+1, 0, 0, 0, 0, 1-p])
        y.append(0)
x_train = x[70:]
x_test = x[:30]
y_train = y[70:]
y_test = y[:30]

# generate the more difficult, less normal dataset
x_hard = []
y_hard = []
for i in range(101):
    if i % 2 == 0:
        p = random.uniform(0.3, 0.4) + random.uniform(-0.1, 0.1)
        x_hard.append([1-p, 0, 0, 0, 0, p+1, -2*p, 0, 0, -2*p, p+1, 0, 0, 0, 0, 1-p])
        y_hard.append(1)
    else:
        p = random.uniform(0.2, 0.3) + random.uniform(-0.1, 0.1)
        x_hard.append([1-p, 0, 0, 0, 0, p+1, -2*p, 0, 0, -2*p, p+1, 0, 0, 0, 0, 1-p])
        y_hard.append(0)
x_train_hard = x_hard[70:]
x_test_hard = x_hard[:30]
y_train_hard = y_hard[70:]
y_test_hard = y_hard[:30]
#print(x_train_hard)
#print(y_train_hard)


# build the normal tree, trained on the normal dataset
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# buid the less normal tree, trained on the more difficult dataset
clf_hard = tree.DecisionTreeClassifier()
clf_hard = clf_hard.fit(x_train_hard, y_train_hard)

accuracy = 0
for i in range(len(y_test)):
    predicted = clf.predict([x_test[i]])[0]
    if predicted == y_test[i]:
        accuracy += 1
accuracy = accuracy / len(y_test)
print("Accuracy for tree 1: {0}".format(accuracy))

accuracy = 0
for i in range(len(y_test_hard)):
    predicted = clf.predict([x_test_hard[i]])[0]
    if predicted == y_test_hard[i]:
        accuracy += 1
accuracy = accuracy / len(y_test)
print("Accuracy for tree 2: {0}".format(accuracy))