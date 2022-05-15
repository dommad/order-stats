import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import scipy
import math
from KDEpy import FFTKDE

class Tools:
    
    def __init__(self):
        pass
    
    #CDFs
    def n_cdf(x, N0, a):

        cdf = np.exp(-N0*np.exp(x/a))
        return cdf

    def n_m1_cdf(x, N0, a):

        cdf = (N0* np.exp(x/a) + 1)*np.exp(-N0*np.exp(x/a))
        return cdf

    def n_m2_cdf(x, N0, a):

        cdf = (N0*N0)/2*np.exp((2*x/a) - N0*np.exp(x/a)) + n_m1_cdf(x, N0, a)
        return cdf

    def n_m3_cdf(x, N0, a):

        cdf = pow(N0, 3)/6*np.exp( (3*x/a) - N0*np.exp(x/a) ) + n_m2_cdf(x, N0, a)
        return cdf

    @staticmethod
    def gumbel_ppf(p, N0, a):
        return a*np.log(-np.log(p)/N0)


    #pdfs

    @staticmethod
    def pdf_universal(x, N0, a, alpha):

        t1 = pow(N0, 1+alpha)
        t2 = math.factorial(alpha)*a
        exp1 = np.exp((1+alpha)*x/a - N0*np.exp(x/a))

        pdf = -(t1/t2)*exp1

        return pdf


    #log-likelihoods    
    
    @staticmethod
    def log_like_universal(log_par, data, alpha):

        N = len(data)
        N0 = np.exp(log_par[0])
        a = -np.exp(log_par[1])
        eq1 = N*np.log(-pow(N0, 1+alpha)/(a*math.factorial(alpha))) + np.sum( ((1+alpha)*data/a) - N0*np.exp(data/a) )

        return eq1

    #MLEs
    
    def mle_universal(self, data, alpha):
        res = scipy.optimize.minimize(
        fun=lambda log_params, data,alpha: -self.log_like_universal(log_params, data, alpha),
        x0=np.array([1, -1]),
        args=(data,alpha,),
           method='BFGS'

        )
        n0, a = np.exp(res.x)
        a = -a
        n0 = n0
        return n0, a
    
    
    #universal pdf, new parametrization
    
    @staticmethod
    def pdf_mubeta(x, mu, beta, alpha):
        beta += 1e-6
        z = (x-mu)/beta
        denom = beta*math.factorial(alpha)
        pdf = (1/denom)*np.exp(-(alpha+1)*z - np.exp(-z))

        return pdf
    
    #log-likelihoods    
    
    @staticmethod
    def log_like_mubeta(log_par, data, alpha):

        N = len(data)
        mu = np.exp(log_par[0])
        beta = np.exp(log_par[1]) + 1e-10
        z  = (data - mu)/beta
        eq1 = -N*np.log(beta*math.factorial(alpha)) - (alpha+1)*np.sum(z) - np.sum(np.exp(-z))

        return eq1
    
    
    def mle_new(self, data, alpha):
        res = scipy.optimize.minimize(
        fun=lambda log_params, data, alpha: -self.log_like_mubeta(log_params, data, alpha),
        x0=np.array([np.log(0.1), np.log(0.02)]),
        args=(data,alpha,),
           method='BFGS'

        )
        mu, beta = np.exp(res.x)
        return mu, beta
    
    @staticmethod
    def gumbel_new_ppf(p, mu, beta):
        
        x = mu - beta*np.log(-np.log(p))

        return x
    
    ### method of moments ###
    
    def mm_estimator(self, data, k):
        data = sorted(data)
        #le = len(data)
        #data = data[int(le*0.05):int(le*0.95)]
        fac = math.factorial(k)
        euler_m = -scipy.special.digamma(1)
        trigamma = math.pi**2/6 - self.sum_squared(k+1)
        m1 = self.first_moment(data)
        m2 = self.second_moment(data)
        
        beta  = np.sqrt((m2 - m1**2 )/trigamma)
        mu = m1 - beta*(euler_m - self.harmonic(k))
        return mu, beta
        
    @staticmethod
    def harmonic(k):
        f = lambda x: 1/x
        squared = list(map(f, np.arange(1, k+1)))
        return np.sum(squared)
    
    @staticmethod
    def first_moment(data):
        return np.mean(data)
    
    @staticmethod
    def second_moment(data):
        f = lambda x: x**2
        squared = list(map(f, data))
        return (1/len(data))*np.sum(squared)
    
    @staticmethod
    def sum_squared(k):
        f = lambda x: 1/x**2
        squared = list(map(f, np.arange(1, k+1)))
        return np.sum(squared)
    
    
    @staticmethod
    def universal_cdf(x, mu, beta, k):
        
        z = (x-mu)/beta
        f = lambda m: np.exp(-m*z)/math.factorial(m)
        summed = 0
        
        for i in range(k):
            #print(f(i))
            summed += f(i)
            
        #print(f"summed is {summed}")
        cdf = np.exp(-np.exp(-z))*summed

        return cdf
    @staticmethod
    def mubeta_cdf(x, mu, beta):
        
        z = (x-mu)/beta
        cdf = np.exp(-np.exp(-z))

        return cdf
    
    """
    def universal_ppf(self, p, alpha, mu, beta):
        
        res = scipy.optimize.minimize(
        fun=lambda log_params, p, mu, beta, alpha: self.universal_cdf(log_params, p, mu, beta, alpha),
        x0=0.1,
        args=(p,mu, beta, alpha,),
           method='BFGS'

        )
        
        return res.x[0]
        """
        
class EM:
    
    def __init__(self):
        self.lows = Tools()
    
    def em_algorithm(self, data, fixed_pars=[]):
 
        #initialization
        left = data[data<0.15]
        right = data[data>=0.15]
        
        old_pi0 = len(left)/len(data)
        old_mu1 = 0.1
        old_beta = 0.02
        old_mu2 = np.mean(right)
        old_sigma = np.std(right)
        
        if old_mu2 < 0.25:
            old_mu2 = 0.25
        print(old_sigma)
        
        pi_j_0 = old_pi0*np.ones(len(data))
        error = 100
        
        fixed = 0        
        if fixed_pars != []:
            old_mu1 = fixed_pars[0]
            old_beta = fixed_pars[1]
            print("fixed = 1")
            fixed = 1
        
        it = 0
        #make sure EM can finish even if it gets stuck in some local optimum
        while (error > 0.0001) and (it < 50):
            
            if fixed == 0:
                new_mu1, new_beta = self.gumbel_params(data, pi_j_0)
            else:
                new_mu1, new_beta = old_mu1, old_beta
                
            new_mu2, new_sigma = self.normal_params(data, 1-pi_j_0)
            
            pi_j_0, new_pi_0 = self.find_pi(data, old_mu1, old_beta, old_mu2, old_sigma, old_pi0)
            
            error = abs(old_pi0-new_pi_0)/old_pi0
            
            old_mu1, old_beta = new_mu1, new_beta
            old_mu2, old_sigma = new_mu2, new_sigma
            old_pi0 = new_pi_0
            #print(error)
            print(old_mu1, old_beta, old_mu2, old_sigma, old_pi0)    
            it += 1
            
        return old_mu1, old_beta, old_mu2, old_sigma, old_pi0


    #formulas to find updated parameters

    def gumbel_params(self, data, p_i):
        
        _, guess_beta = self.lows.mm_estimator(data, 0)
        #print(guess_beta)
        
        new_beta = scipy.optimize.newton(self.find_beta, x0=guess_beta, args=(data, p_i))
        new_mu = new_beta*np.log(np.sum(p_i)/np.sum(np.exp(-data/new_beta)*p_i))
        return new_mu, new_beta
        
    @staticmethod
    def find_beta(beta, x, p_i):
        
        mu = beta*np.log(np.sum(p_i)/np.sum(np.exp(-x/beta)*p_i))
        return beta*np.sum(p_i) + np.sum( (x-mu)*(np.exp((mu-x)/beta)-1)*p_i)

    @staticmethod
    def normal_params(data, p_i):
        
        new_mu = np.sum(data*p_i)/np.sum(p_i)
        new_var = np.sum(np.power(data-new_mu, 2)*p_i)/np.sum(p_i)
        
        return new_mu, np.sqrt(new_var)


    #formula to find updated pi_0

    @staticmethod
    def find_pi(data, mu_0, beta_0, mu_1, sigma_1, pi_0_old):
        
        f_0 = st.gumbel_r.pdf(data, mu_0, beta_0)*pi_0_old
        f_1 = st.norm.pdf(data, mu_1, sigma_1)*(1-pi_0_old)
        
        pi_j_0 = f_0/(f_0 + f_1)
        #pi_j_1 = 1 - pi_j_0
        
        pi_0_new = np.mean(pi_j_0)
        #pi_1_new = np.mean(pi_j_1)
        
        return pi_j_0, pi_0_new
    
    @staticmethod
    def plot_em(ax, data, params):
        
        axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
        normed = auc(axes, kde)
        kde0 = st.gumbel_r.pdf(axes, params[0], params[1])
        kde1 = st.norm.pdf(axes, params[2], params[3])

        #fig, ax = plt.subplots(figsize=(6,6))

        ax.fill_between(axes, kde/normed, color='green', alpha=0.3)
        ax.plot(axes, kde/normed, color='green')
        ax.plot(axes, params[4]*kde0, color='red')
        ax.plot(axes, (1-params[4])*kde1, color='blue')
        ax.set_ylim(0,)
        ax.set_xlabel("TEV")
        ax.set_ylabel("density")
        
        #fig.tight_layout()
        #fig.savefig(f"./graphs/{outname}.png", dpi=600, bbox_inches='tight')
            
    
    
    
    
        
        
    
    
    
