import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import util

def create_mesh_vals(x, y):
    ret = []
    for j in y:
        for i in x:
            ret.append([i, j])
    return np.array(ret)

fig_num = 0
def plot_guassian(mu, cov, xlabel, ylabel, title, window_l=-1, window_u=1):
    global fig_num
    x = np.arange(-1, 1.1, step=0.01)
    y = np.arange(-1, 1.1, step=0.01)
    xx, yy = np.meshgrid(x, y)
    #print(x)
    z = create_mesh_vals(x, y)
    #print(z[21*5+9])

    pa = util.density_Gaussian(mu, cov, z).reshape((210, 210))
    #print(c2)

    plt.figure(fig_num)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cp = plt.contour(xx, yy, pa)
    plt.xlim((window_l, window_u))
    plt.ylim((window_l, window_u))

    fig_num += 1

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    mu = np.array([0,0])
    cov = np.array([[beta, 0],[0, beta]])
    plot_guassian(mu, cov, "a0", "a1", "distribution of p(a)")
    plt.scatter(-0.1, -0.5)
    
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    size = x.shape[0]

    x_aux = np.concatenate((np.ones((size, 1)), x), axis=1)
    c = x_aux.T@x_aux + (sigma2/beta)*np.identity(2, dtype=float)

    mu = inv(c)@x_aux.T@z
    mu = mu.flatten()

    cov = inv(c)*sigma2

    return (mu, cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    mu = mu.reshape((2, 1))

    size = x.shape[0]
    x_aux = np.concatenate((np.ones((size, 1)), x), axis=1)
    #the new z values
    mu_new = mu.T@x_aux.T

    #print(Cov.shape, x_aux.shape)
    cov_new = x_aux@Cov@x_aux.T
    #print(cov_new.shape

    
    global fig_num

    plt.figure(fig_num)
    plt.title("predictions with "+ str(x_train.shape[0]) +" examples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x_train, z_train)
    plt.errorbar(x.flatten(), mu_new.flatten(), np.sqrt(cov_new.diagonal()))
    #cp = plt.contour(xx, yy, pa)
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))

    fig_num += 1

    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1

    # prior distribution p(a)
    priorDistribution(beta)

    d_sizes = [1, 5, 100]
    for sizes in d_sizes:
        # number of training samples used to compute posterior
        ns = sizes
        
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]
        
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        plot_guassian(mu, Cov, "a0", "a1", "distribution of p(a|x,z) with \n" + str(sizes) + " examples")
        plt.scatter(-0.1, -0.5)

        # distribution of the prediction
        v = np.arange(-4, 4, step=0.2)
        v = v.reshape((v.shape[0], 1))
        predictionDistribution(v, beta, sigma2, mu,Cov, x, z)
    plt.show()
    

   

    
    
    

    
