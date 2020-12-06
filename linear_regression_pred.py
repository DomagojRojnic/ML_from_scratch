# Predavanja primjeri
# Odredivanje parametara linearnog modela pomocu metode gradijentnog spusta i izravnog rjesenja
# model je oblika: y_p = theta_1 * x + theta_0
# R.Grbic

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model as lm

def pravac(x, theta):

    return theta[1]*x + theta[0]

def kriterijJ(x,y,theta):
    J = 0.0
    n = x.shape[0]
    for i in range(0,n):
        J += (pravac(x[i],theta) - y[i]) ** 2
    
    J /= (2*n)

    return J
        


if __name__ == "__main__":
    n_samples = 100
    n_outliers = 10


    # simulacijski podaci
    x, y = datasets.make_regression(n_samples=n_samples,n_features=1,
                                    n_informative=1, noise=10.0, coef=False,
                                    random_state=0)

    #Add outlier data
    np.random.seed(0)
    x[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
    y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

    #prikazi podatke
    plt.figure(1)
    plt.title("Podaci")
    plt.scatter(x,y,marker='.')
 
    #izravno rjesenje
    X = np.ones((n_samples,1))
    X = np.append(X, x, axis=1)
    theta_direct = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    print("Theta0 direktno rjesenje: ", theta_direct[0])
    print("Theta1 direktno rjesenje: ", theta_direct[1])
    print("Vrijednost kriterijske funkcije: ", kriterijJ(x,y,theta_direct))

    #prikazi pravac
    xp = np.array([x.min(), x.max()])
    yp = np.array([pravac(xp[0],theta_direct), pravac(xp[1],theta_direct)])

    plt.plot(xp,yp,'r')
    plt.show()


    # rjesenje pomocu scikitlearn
    linearniModel = lm.LinearRegression()
    linearniModel.fit(x, y)
    print("Sckit theta0: ", linearniModel.intercept_)
    print("Sckit theta1: ", linearniModel.coef_)


    # metoda najbrzeg spusta
    no_iter = 200
    theta_old = np.zeros((2,1))
    theta_new = np.zeros((2,1))
    dulj_koraka = 0.1
    J = np.zeros((no_iter,1))

    plt.figure(2)
    plt.scatter(x, y, marker='.')   
    plt.title("gradient descent")

    for iter in range(0, no_iter):

        J[iter] = kriterijJ(x,y,theta_old)
        
        rj0 = 0.0
        rj1 = 0.0

        for i in range(0,n_samples):
            rj0 += pravac(x[i],theta_old) - y[i]
            rj1 += (pravac(x[i],theta_old) - y[i])*x[i]

        rj0 /= n_samples
        rj1 /= n_samples

        theta_new[0] = theta_old[0] - dulj_koraka * rj0
        theta_new[1] = theta_old[1] - dulj_koraka * rj1
        theta_old = theta_new

        yp[0] = pravac(xp[0],theta_new)
        yp[1] = pravac(xp[1],theta_new)
        
        plt.plot(xp,yp)
        plt.pause(0.05)
    

    plt.figure(3)
    plt.plot(range(iter+1),J)
    plt.ylabel('Vrijednost kriterijske funkcije')
    plt.xlabel('iteracija')
    plt.show()





    




