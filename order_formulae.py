"""Distribution functions, parameter estimation codes"""
from math import pi as PI
from math import factorial as fact
from functools import reduce
import scipy.stats as st
import numpy as np
from sklearn.metrics import auc
from scipy.optimize import minimize, newton
from scipy.special import digamma, comb
from KDEpy import FFTKDE

class Tools:
    """maximum likelihood estimation and method of moments
    for lower order statistics"""

    def __init__(self):
        pass

    @staticmethod
    def pdf_mubeta(x_val, mu_, beta, alpha):
        """PDF for TEV distribution of order alpha, mu-beta parametrization"""

        beta += 1e-6
        z_val = (x_val-mu_)/beta
        denom = beta*fact(alpha)
        pdf = (1/denom)*np.exp(-(alpha+1)*z_val - np.exp(-z_val))

        return pdf


    @staticmethod
    def log_like_mubeta(log_par, data, alpha):
        """log-likelihood function with mu-beta parametrization"""

        no_points = len(data)
        mu_, beta = np.exp(log_par) + 1e-10
        z_var  = (data - mu_)/beta
        fac = fact(alpha)
        eq1 = -no_points*np.log(beta*fac) - (alpha+1)*np.sum(z_var) - np.sum(np.exp(-z_var))

        return eq1


    def mle_mubeta(self, data, alpha):
        """MLE using mu-beta parametrization for model of order alpha"""

        res = minimize(
        fun=lambda log_params, data, alpha: -self.log_like_mubeta(log_params, data, alpha),
        x0=np.array([np.log(0.1), np.log(0.02)]),
        args=(data,alpha,),
           method='Nelder-Mead')
        mu_, beta = np.exp(res.x)

        return mu_, beta

    # functions for simulation of estimation for finite N distribution forms

    @staticmethod
    def log_like_fin_n(log_par, data, alpha, no_c):
        """log-likelihood for finite N form, N is the same for all data points"""

        no_ = len(data)
        alpha += 1
        mu_, beta = np.exp(log_par) + 1e-10
        z_var  = ((data - mu_)/beta)
        cdf_d = 1 - (1/no_c)*np.exp(-z_var)
        pdf_d = 1/(no_c*beta)*np.exp(-z_var)
        re_ = no_c-alpha+1
        te1 = no_*np.log(re_*comb(no_c, re_))
        te2 = np.sum(np.log(pdf_d)) + (re_-1)*np.sum(np.log(cdf_d))
        te3 = (no_c-re_)*np.sum(np.log(1-cdf_d))
        eq1 = te1 + te2 + te3

        return -eq1

    def mle_fin_n(self, data, alpha, no_c):
        """MLE using mu-beta parametrization for model of order alpha
        no_c: number of candidates scored for each spectrum"""

        res = minimize(
        fun = self.log_like_fin_n,
        x0=np.array([np.log(0.1), np.log(0.02)]),
        args=(data, alpha, no_c),
        method='Nelder-Mead')

        mu_, beta = np.exp(res.x)

        return mu_, beta



    @staticmethod
    def gumbel_new_ppf(pvalue, mu_, beta):
        """quantile function for Gumbel distribution"""
        return mu_ - beta*np.log(-np.log(pvalue))


    ### METHOD OF MOMENTS ###

    def mm_estimator(self, data, k):
        """MM estimates of mu and beta for order k"""

        data = sorted(data)
        euler_m = -digamma(1)
        m_1 = self.first_moment(data)
        m_2 = self.second_moment(data)
        beta  = np.sqrt((m_2 - m_1**2 )/self.trigamma(k))
        mu_ = m_1 - beta*(euler_m - self.harmonic(k))

        return mu_, beta


    def trigamma(self, k):
        """trigamma function"""
        return PI**2/6 - self.sum_squared(k+1)


    @staticmethod
    def harmonic(k):
        """harmonic number"""
        squared = list(map(lambda x: 1/x, np.arange(1, k+1)))
        return np.sum(squared)


    @staticmethod
    def first_moment(data):
        """raw first moment"""
        return np.mean(data)


    @staticmethod
    def second_moment(data):
        """raw second moment"""
        squared = list(map(lambda x: x**2, data))
        length = len(data)
        if length == 0:
            length = 1
        return (1/length)*np.sum(squared)


    @staticmethod
    def sum_squared(k):
        """sum of squared values"""
        squared = list(map(lambda x: 1/x**2, np.arange(1, k+1)))
        return np.sum(squared)


    @staticmethod
    def universal_cdf(x_val, mu_, beta, k):
        """CDF of TEV distribution of order k for mu-beta parametrization"""
        z_val = (np.array(x_val)-mu_)/beta
        factorials = map(lambda m: np.exp(-m*z_val)/fact(m), range(k+1))
        summed = reduce(lambda x,y: x+y, factorials)
        cdf = np.exp(-np.exp(-z_val))*summed
        return cdf

    @staticmethod
    def cdf_finite_n(x_val, mu_, beta, k, n_cand):
        """CDF of TEV distribution of order k for finite N form"""
        z_val = (np.array(x_val)-mu_)/beta
        fx_ = 1 - (1/n_cand)*np.exp(-z_val)
        summands = np.zeros(len(z_val))
        for idx in range(k+1):
            cur_sum = comb(n_cand, n_cand-idx)*pow(fx_, n_cand-idx)*pow(1-fx_, idx)
            summands += cur_sum
        return summands


class EM:
    """expectation-maximization for two-components mixture: Gumbel + Gaussian"""

    def __init__(self):
        self.lows = Tools()

    def em_algorithm(self, data, fixed_pars=()):
        """Executes EM algorithm for the data provided"""

        # initialization
        left = data[data<0.15]
        right = data[data>=0.15]

        old_pi0 = len(left)/len(data)
        old_mu1 = 0.1
        old_beta = 0.02
        old_mu2 = np.mean(right)
        old_sigma = np.std(right)
        old_mu2 = max(old_mu2, 0.25)

        pi_j_0 = old_pi0*np.ones(len(data))
        error = 100
        fixed = 0

        if fixed_pars != ():
            old_mu1 = fixed_pars[0]
            old_beta = fixed_pars[1]
            print("fixed = 1")
            fixed = 1

        iteration = 0
        # make sure EM can finish even if it gets stuck in a local optimum
        while (error > 0.0001) and (iteration < 50):

            if fixed == 0:
                new_mu1, new_beta = self.__gumbel_params(data, pi_j_0)
            else:
                new_mu1, new_beta = old_mu1, old_beta

            new_mu2, new_sigma = self.__normal_params(data, 1-pi_j_0)
            pi_j_0, new_pi_0 = self.__find_pi(data, old_mu1, old_beta, old_mu2, old_sigma, old_pi0)

            error = abs(old_pi0-new_pi_0)/old_pi0

            old_mu1, old_beta = new_mu1, new_beta
            old_mu2, old_sigma = new_mu2, new_sigma
            old_pi0 = new_pi_0

            print(old_mu1, old_beta, old_mu2, old_sigma, old_pi0)

            iteration += 1

        return old_mu1, old_beta, old_mu2, old_sigma, old_pi0


    # formulae to update the parameters

    def __gumbel_params(self, data, p_i):
        """get Gumbel parameters"""

        _, guess_beta = self.lows.mm_estimator(data, 0)
        #print(guess_beta)

        new_beta = newton(self.__find_beta, x0=guess_beta, args=(data, p_i))
        new_mu = new_beta*np.log(np.sum(p_i)/np.sum(np.exp(-data/new_beta)*p_i))
        return new_mu, new_beta

    @staticmethod
    def __find_beta(beta, x_val, p_i):
        """find beta"""
        mu_ = beta*np.log(np.sum(p_i)/np.sum(np.exp(-x_val/beta)*p_i))
        return beta*np.sum(p_i) + np.sum( (x_val-mu_)*(np.exp((mu_-x_val)/beta)-1)*p_i)

    @staticmethod
    def __normal_params(data, p_i):
        """new normal parameters"""
        new_mu = np.sum(data*p_i)/np.sum(p_i)
        new_var = np.sum(np.power(data-new_mu, 2)*p_i)/np.sum(p_i)
        return new_mu, np.sqrt(new_var)


    @staticmethod
    def __find_pi(data, mu_0, beta_0, mu_1, sigma_1, pi_0_old):
        """find the updated pi_0"""

        f_0 = st.gumbel_r.pdf(data, mu_0, beta_0)*pi_0_old
        f_1 = st.norm.pdf(data, mu_1, sigma_1)*(1-pi_0_old)

        pi_j_0 = f_0/(f_0 + f_1)
        #pi_j_1 = 1 - pi_j_0
        pi_0_new = np.mean(pi_j_0)
        #pi_1_new = np.mean(pi_j_1)

        return pi_j_0, pi_0_new


    @staticmethod
    def plot_em(axs, data, params):
        """plot the results of expectation-maximization algorithm"""

        axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
        normed = auc(axes, kde)
        kde0 = st.gumbel_r.pdf(axes, params[0], params[1])
        kde1 = st.norm.pdf(axes, params[2], params[3])

        #fig, ax = plt.subplots(figsize=(6,6))

        axs.fill_between(axes, kde/normed, color='green', alpha=0.3)
        axs.plot(axes, kde/normed, color='green')
        axs.plot(axes, params[4]*kde0, color='red')
        axs.plot(axes, (1-params[4])*kde1, color='blue')
        axs.set_ylim(0,)
        axs.set_xlabel("TEV")
        axs.set_ylabel("density")

        #fig.tight_layout()
        #fig.savefig(f"./graphs/{outname}.png", dpi=600, bbox_inches='tight')
