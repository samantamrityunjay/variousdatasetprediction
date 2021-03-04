import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

X,y = datasets.make_classification(n_samples=1000, n_features=2, shuffle =True, n_redundant=0, )

print(X.shape)
print(X[:10,:])

print(y.shape)
print(y[:10])

# plt.scatter(X[:,0],X[:,1], c=y)
# plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2021)
# plt.scatter(X_train[:,0],X_train[:,1], c=y_train)
# plt.show()
# plt.scatter(X_test[:,0],X_test[:,1], c=y_test)
# plt.show()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
def accuracy(true,predict):
    return np.sum(true==predict)/len(true)
print(accuracy(y_test,y_predict))
print(clf.score(X_test,y_test))

print(clf.coef_)
print(clf.classes_)
print(clf.intercept_)
x_min,x_max = X_train[:,0].min(),X_train[:,0].max()
y_min,y_max = -x_min*clf.coef_[:,0]/clf.coef_[:,1] - clf.intercept_/clf.coef_[:,1] , -x_max*clf.coef_[:,0]/clf.coef_[:,1] - clf.intercept_/clf.coef_[:,1]
plt.plot([x_min,x_max],[y_min,y_max])
plt.scatter(X_train[:,0],X_train[:,1], c=y_train)
plt.show()