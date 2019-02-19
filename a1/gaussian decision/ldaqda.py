import numpy as np
#import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    x_m = x[np.where(y == 1)[0]]
    x_f = x[np.where(y == 2)[0]]
    # print(np.count_nonzero(y == 1))
    # print(np.count_nonzero(y == 2))
    # print(np.where(y == 1)[0].shape)
    # print(np.where(y == 2)[0].shape)

    mu = np.mean(x, axis=0)
    mu_male = np.mean(x_m, axis=0)
    mu_female = np.mean(x_f, axis=0)

    N = x.shape[0]
    N_male = x_m.shape[0]
    #print(N_male)
    N_female = x_f.shape[0]
    #print(N_female)
    #print(np.transpose(x_m - mu_male))
    #print((x_m - mu_male))
    #print(np.transpose(x_m - mu_male)@(x_m - mu_male))
    cov = np.transpose(x - mu)@(x - mu)*(1/N)
    cov_male = np.transpose(x_m - mu_male)@(x_m - mu_male)*(1/N_male)
    cov_female = np.transpose(x_f - mu_female)@(x_f - mu_female)*(1/N_female)

    #print(mu_male, mu_female,cov,cov_male,cov_female)
    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def get_lda(mu, cov, x):
    cov_inv = np.linalg.inv(cov)
    mu_t = np.transpose((mu))
    return  x@cov_inv@mu_t - (1/2)*mu_t@cov_inv@mu + np.log(0.5)

def get_qda(mu, cov, x):
    cov_inv = np.linalg.inv(cov)
    cov_mag = np.linalg.norm(cov)**(1/2)
    #print(cov_mag)

    y = (x-mu)
    #y_t = np.transpose(y)
    #print(mu)
    #print(x)
    #print(y)
    #print(cov_inv)
    #print((y@cov_inv))
    #print((y@cov_inv)*y)
    #return get_lda(mu, cov, x) - (1/2)*np.sum((x@cov_inv)*x, axis = 1) - np.log(cov_mag)
    return -1*(1/2)*np.sum((y@cov_inv)*y, axis=1) - np.log(cov_mag) + np.log(0.5)

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    N = x.shape[0]
    lda_males = get_lda(mu_male, cov, x)
    lda_females = get_lda(mu_female, cov, x)
    c_lda_predict = (lda_males > lda_females) == (y == 1)
    #print(c_lda_predict)
    #print(cov_male)
    #print(cov_female)
    qda_males = get_qda(mu_male, cov_male, x)
    qda_females = get_qda(mu_female, cov_female, x)
    # print((qda_males > qda_females))
    # print((y == 1))
    c_qda_predict = (qda_males > qda_females) == (y == 1)
    print(c_qda_predict)

    mis_lda = (N - np.count_nonzero(c_lda_predict))/N
    mis_qda = (N - np.count_nonzero(c_qda_predict))/N
    print(mis_lda, mis_qda)
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
