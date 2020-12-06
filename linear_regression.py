import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def hipothesis(x, theta):
    return theta[0] + theta[1] * x

def cost_function(X, Y, theta):
    J = 0.0
    m = X.shape[0]
    for i in range(0, m):
        J += ( hipothesis(X[i], theta) - Y[i]) ** 2
    return J/(2*m)

data = pd.read_csv('student_scores.csv')
X = pd.DataFrame(data.iloc[:,0], columns = ['Hours'])
Y = pd.DataFrame(data.iloc[:,1], columns = ['Scores'])

iterations = 200
learning_rate = 0.1
theta_old = np.zeros((2,1))
theta_new = np.zeros((2,1))
J = np.zeros((iterations, 1))

xp = np.array([X.min(), X.max()])
yp = np.array( [hipothesis(xp[0], theta_old), hipothesis(xp[1], theta_old)] )

plt.figure(1)
plt.title('Gradient descent')
plt.scatter(X, Y, marker='.')
plt.show()

for iteration in range(0, iterations):

    J[iteration] = cost_function(X, Y, theta_old)

    temp0 = 0.0
    temp1 = 0.0

    for i in range(0, X.shape[0]):
        temp0 += (hipothesis(X[i], theta_old) - Y[i])
        temp1 += (hipothesis(X[i], theta_old) - Y[i]) * X[i]
    
    theta_new[0] = theta_old[0] - learning_rate * (temp0/X.shape[0])
    theta_new[1] = theta_old[0] - leaning_rate * (temp1/X.shape[0])

    theta_old = theta_new

    # yp[0] = hipothesis(xp[0],theta_new)
    # yp[1] = hipothesis(xp[1],theta_new)
        
    plt.plot()
    plt.pause(0.05)

plt.figure(2)
plt.plot(iterations, J)
plt.show()