import numpy as np
from numpy.random import multivariate_normal, normal
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class ES:
    
    def __init__(self, n_dims:int, cost_func):
        self.n_dims = n_dims
        self.cost_func = cost_func
        self.ans = np.random.normal(0,1,n_dims)
        self.sigmas = np.zeros(n_dims)
        self.alphas = np.zeros(int(n_dims*(n_dims-1)/2))
        self.cost = self.cost_func(self.ans)
        self.cost_history = []
    

    def evolve(self, iterations:int=1000, lru:int=None, lrds:list=None, sigma_lb:float=0.0001, beta:float=0.5
        , consider_cov=False):
        """
        Parameters:
        iterations (int): maximun number of iterations to evolve
        lrds (float): learning rate for dimentions
        lru (float): universal learning rate
        beta (float): learning rate for angles
        sigma_lb (float): sigma lower bound
        consider_cov (bool): should consider covariance between each dimensions or not
        """

        logging.debug("setting lrds and lru")
        if not lrds:
            lrds = [1/np.sqrt(2*np.sqrt(self.n_dims))]*self.n_dims
        if not lru:
            lru = 1/np.sqrt(2*self.n_dims)

        # for generating new samples with mean 0
        means = np.zeros(self.n_dims) 

        for it in range(iterations):
            if it % 10 == 0:
                logging.info('iteration {}'.format(it))

            # mutate sigmas
            logging.debug('mutating sigmas')
            tsigmas = np.zeros(self.n_dims)
            for i in range(self.n_dims):
                sigma = self.sigmas[i] * np.exp(lru*normal(0,1)+lrds[i]*normal(0,1))
                if sigma < sigma_lb:
                    sigma = sigma_lb
                tsigmas[i] = sigma

            #mutate alphas
            talphas = None
            if consider_cov:
                logging.debug('mutating alphas')
                talphas = self.alphas+beta*normal(0,1, len(self.alphas))
            
            logging.debug('building covariance matrix')
            cov_matrix = self.make_cov_matrix(tsigmas, talphas, consider_cov)

            logging.debug('generating new answer')
            tans = self.ans +  multivariate_normal(means, cov_matrix)

            logging.debug('calculating cost of new answer')
            tans_cost = self.cost_func(tans)
            
            if  tans_cost < self.cost:
                logging.info('cost improved from {:.5f} to {:.5f}'.format(self.cost, tans_cost))
                self.ans = tans
                self.sigmas = tsigmas
                self.alphas = talphas
                self.cost = tans_cost
                if tans_cost == 0.0:
                    break
            else:
                logging.info('cost not improved from {:.5f}'.format(self.cost))
            
            self.cost_history.append(self.cost)


        logging.info('best solution found:{}'.format(self.ans))



    def make_cov_matrix(self, sigmas, alphas, consider_cov):
        
        logging.debug('make upper triangular convariance matrix')
        cov_matrix = np.diagflat(sigmas)

        if consider_cov:
            logging.debug('make upper triangular matrix from alphas')
            tri_alphas = np.zeros((self.n_dims, self.n_dims))
            indices = np.triu_indices(self.n_dims, 1)
            tri_alphas[indices] = alphas
        
            for i in range(self.n_dims):
                for j in range(self.n_dims):
                    if j > i:
                        cov_matrix[i,j] = 1/2*(cov_matrix[i][i]-cov_matrix[j][j])* np.tan(2*tri_alphas[i,j])  
            logging.debug('copy upper triangle to lower triangle')
            indices = np.triu_indices(self.n_dims, -1)
            cov_matrix[indices] = cov_matrix.T[indices]

        return cov_matrix
    

    def plot_history(self, path='plt.png'):
        plt.figure(1, dpi=300)
        x = list(range(len(self.cost_history)))
        plt.plot(x, self.cost_history)
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.savefig(path)
        # plt.show()
        # plt.pause(1)
        