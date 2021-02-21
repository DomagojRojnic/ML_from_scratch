# Predavanja primjeri
# Logisticka regresija
# R.Grbic

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression


def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))


def predict(x,theta):

    z = x @ theta
    return sigmoid(z)


def costFunction(x,y,theta):

    n = len(y)

    h = predict(x, theta)

    # kada je y jednak 1
    J1 = y*np.log(h)

    # kada je y jednak 2
    J2 = (1-y)*np.log(1-h)

    J = J1.sum() + J2.sum()
    J *= -1.0
    J /= n

    return J

def probabilityToClass(h):

    return 1*(h >= 0.5)


def accuracy(X, y, theta):

    n = len(y)

    h = predict(X, theta)

    y_p = probabilityToClass(h)

    accuracy = 100.0*(n - sum(abs(y-y_p))) / n

    return accuracy




# generiraj 2d klasifikacijski problem
n = 200
X, y = make_blobs(n_samples=n, centers=2, n_features=2, cluster_std=1.2, random_state=4)
y = np.reshape(y,(n,1))
X = np.append(np.ones((n,1)), X, axis=1)


# prikazi podatke
plt.figure(1)
plt.title("Logisticka regresija")
plt.scatter(X[:,1], X[:,2], c=y)
plt.xlabel('x_1')
plt.ylabel('x_2')
#plt.show()

# metoda gradijentnog spusta
no_iter = 300000
learning_rate = 0.05
theta = np.zeros((3,1))
J = np.zeros((no_iter,1))

for i in range(0,no_iter):

    # h(x)
    h = predict(X, theta)
    gradijent = X.T @ (h-y)
    gradijent /= n
    theta = theta - learning_rate * gradijent
    
    J[i] = costFunction(X,y,theta)
    
    # log
    if i % 100 == 0:
        print("Iteracija: "+str(i) + "  J: "+str(J[i]))

print("Vrijednosti parametara logisticke regresije dobiveni metodom gradijentnog spusta")
print("theta_0: " + str(theta[0]))
print("theta_1: " + str(theta[1]))
print("theta_2: " + str(theta[2]))


# prikazi granicu odluke
xp = np.array([X[:,1].min(), X[:,1].max()])
yp1 = -theta[1]/theta[2] * xp[0] - theta[0]/theta[2]
yp2 = -theta[1]/theta[2] * xp[1] - theta[0]/theta[2]
yp = np.array([yp1,yp2])
plt.plot(xp,yp,'r')
plt.show()

plt.figure(2)
plt.plot(range(no_iter),J)
plt.title('Vrijednost kriterijske funkcije')
plt.ylabel('J [i]')
plt.xlabel('iteracija')
plt.show()

# tocnost klasifikatora
acc = accuracy(X,y,theta)
print("Tocnost klasifikatora: " + str(acc[0]) + "%")

# scikit learn logisticka regresija;bez regularizacije
logreg = LogisticRegression(fit_intercept=True, C = 100000000)
logreg.fit(X[:,1:],np.ravel(y))
a = logreg.coef_
print("Parametri logistice regresije dobiveni pomocu ugradene funkcije u sklearn")
print("theta_0: " + str(logreg.intercept_[0]))
print("theta_1: " + str(logreg.coef_[0][0]))
print("theta_1: " + str(logreg.coef_[0][1]))

# prikazi granicu odluke
xp = np.array([X[:,1].min(), X[:,1].max()])
yp1 = -logreg.coef_[0][0]/logreg.coef_[0][1] * xp[0] - logreg.intercept_[0]/logreg.coef_[0][1]
yp2 = -logreg.coef_[0][0]/logreg.coef_[0][1] * xp[1] - logreg.intercept_[0]/logreg.coef_[0][1]
yp = np.array([yp1,yp2])
plt.figure(3)
plt.title("Scikit learn logisticka regresija")
plt.scatter(X[:,1], X[:,2], c=y)
plt.plot(xp,yp,'g')
plt.show()