import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from scipy.optimize import fsolve
import scipy as sc
import math
from pyteomics import pepxml
import random
import os
import glob
from collections import deque
import pickle
import importlib as imp
import lower as low
imp.reload(low)
imp.reload(plt)
lows = low.Tools()


#preprocessing of the data

class Estimation:
        
    def __init__(self, paths):
        
        self.tev = deque()
        self.charges = deque()
        self.big_n = deque()
        
        #parse the pepxml data
        self.parse_pepxmls(paths)
        
    def main_task(self):
        
        params = np.zeros((7,2))
        
        for ch in [2,3,4]:
            a, b = self.test_it(ch)
            if ch == 2:
                params[0,:] = a,b
                params[1,:] = a,b
            if ch == 3:
                params[2,:] = a,b
            if ch == 4:
                params[3,:] = a,b
                params[4,:] = a,b
                params[5,:] = a,b
                params[6,:] = a,b
                
            df = pd.DataFrame(params)
            df.to_csv("pars.txt", sep=' ', index= False, header=False)
        
        
    def test_it(self, char):
        
        tevs, mu, beta, linreg = self.linear_regression(char)
        #self.plot_params(mu[4:], beta[4:], outname = f'{char}_gm')
        opt_mu, opt_beta = self.get_quantiles(tevs[:, 0], linreg, 'testing_now')
        
        #self.plot_best_fit(tevs[:,0], opt_mu, opt_beta)
        idx=0
        #self.plot_fitted(tevs[:, idx], opt_mu , opt_beta, idx, col='blue', frac=0.5, bins=200)
        mean = opt_mu - 0.57721*opt_beta
        sd = math.pi/np.sqrt(6)*opt_beta
        
        return [mean,sd]
        
        
        
    #input: charge, output: LR params 
    def linear_regression(self, charge):
        
        tevs, bg = self.filter_charge(self.tev, self.charges, self.big_n, charge)
        mu, beta = self.lower_params(tevs)
        trim_mu = list(mu)[4:]
        trim_beta = list(beta)[4:]
        linreg = st.linregress(trim_mu, trim_beta)
        print(linreg)
    
        return tevs, mu, beta, linreg
    
    
    #quantile optimization
    def get_quantiles(self, tev, linreg, outname, mode="MLE", cutoff=0.15):
    
        emps = sorted(tev)
        emps_df = pd.DataFrame(emps)
    
        emps_df[1] = emps_df.index + 1
        emps_df[2] = emps_df[1]/len(emps)
    
        #quantile optimization
        errors = []
        qq_range = np.linspace(0.05, 0.4, 500)
        
        if mode == "MLE":
            opt_a = np.mean(tev[-3:])
            for i in qq_range:
                #cur_a = i*linreg.slope + linreg.intercept
                cur_a = opt_a
                theor_q = lows.gumbel_new_ppf(emps_df[2].to_numpy()[:-1], i, cur_a)
                length = len(emps_df[2][emps_df[2] < cutoff])
                
                diffs = theor_q - emps_df[0].to_numpy()[:-1]
                diffs = abs(np.sum(diffs[:length]))
                diffs = diffs/len(theor_q[:length])
                errors.append(diffs)
            
        elif mode == "MM":
            opt_a = np.mean(mms[:,1][-1:])
            for i in qq_range:
       
                cur_a = opt_a
                theor_q = lows.gumbel_new_ppf(emps_df[2].to_numpy()[:-1], i, cur_a)
                length = len(emps_df[2][emps_df[2] < cutoff])
          
                diffs = theor_q - emps_df[0].to_numpy()[:-1]
                diffs = abs(np.sum(diffs[:length]))
                diffs = diffs/len(theor_q[:length])
                errors.append(diffs)
            
        else:
            print("Incorrect mode!")
            return
        
        
    
        opt_idx = errors.index(min(errors))
        opt_N0 = qq_range[opt_idx]
        #opt_a = opt_N0*linreg.slope + linreg.intercept
        
        
        #plotting
        fig = plt.figure(figsize=(6,6))
        plt.semilogy(qq_range, errors)
        plt.scatter(qq_range, errors, s=1)
        #plt.xlim(60,70)
        plt.xlabel("mu")
        plt.ylabel("loss")
        plt.savefig(f"./graphs/1{outname}.png", dpi=600, bbox_inches='tight')
    
        theor_q = lows.gumbel_new_ppf(emps_df[2].to_numpy()[:-1], opt_N0, opt_a)
        emps_q = emps_df[0].to_numpy()[:-1]
    
        fig1 = plt.figure(figsize=(6,6))
        plt.plot([0, 1], [0, 1], color='k')
        plt.scatter(theor_q, emps_q, s=2)
    
        plt.xlim(0, 0.2)
        plt.ylim(0, 0.2)
        plt.xlabel("theoretical quantile")
        plt.ylabel('empirical quantile')
        plt.savefig(f"./graphs/2{outname}.png", dpi=600, bbox_inches='tight')
        print(opt_N0, opt_a)
        
        
        return opt_N0, opt_a
    
        
        
    def plot_best_fit(self, tevs, best_mu, best_beta):
        
        self.plot_fitted(tevs, best_mu , best_beta, 0, frac=0.99, bins=200)
    
    
    
    
    #input: pepxml files, output: np arrays (TEV score, charge, N)
    def parse_pepxmls(self, paths):
        
        for path in paths:
            print(path)
            cur_file = pepxml.read(path)
    
            for spec in cur_file:
                if 'search_hit' in spec.keys():
                    if len(spec['search_hit']) >= 10:
                        self.tev.append(list(map(lambda x: -0.02*np.log(spec['search_hit'][x]['search_score']['expect']/1000), np.arange(10))))
                        self.charges.append(int(spec['assumed_charge']))
                        self.big_n.append(int(spec['search_hit'][0]['num_matched_peptides']))
    
    
    #get only tevs of selected charge
    @staticmethod
    def filter_charge(tev, charges, big_n, ch):
        t = np.array(tev)
        c = np.array(charges)
        n = np.array(big_n)
        mask = np.where((c == ch))
        return t[mask], n[mask]
    
    
    #1 objective 1: estimate parameters for each hit separately, then plot the linear regression
    @staticmethod
    def lower_params(arr):
        n0 = []
        a = []
    
        for hit in range(10):
            #if hit == 0: continue
            #print(hit)
            cur_tev = arr[:,hit].astype('float128')
            cur_n0, cur_a = lows.mle_new(cur_tev, hit)
            n0.append(cur_n0)
            a.append(cur_a)
        return n0, a
    
    
    ######### visualization of the results ###################
    
    def plot_fitted(self, arr, N0, a, alpha, col='blue', frac=1, bins=500, outname=""):
        sorted_arr = np.array(sorted(arr))
        l_lim = sorted_arr[0]
        u_lim = sorted_arr[-1]
        pdf = lows.pdf_mubeta(sorted_arr, N0, a, alpha)
        fig = plt.figure(figsize=(5,5))
        plt.plot(sorted_arr, frac*pdf,color=col)
        sns.distplot(sorted_arr, bins = np.linspace(0, 0.8, bins), kde=False, norm_hist=True,
                    hist_kws=dict(histtype='step', linewidth=1, color='black'))
        plt.xlim(l_lim, u_lim)
        if outname != "":
            fig.savefig(f'./graphs/fitted_{outname}.png', bbox_inches='tight', dpi=400)
    
    
    def plot_params(self, n0, a, outname = ""):
    
        trim_n0 = list(n0)
        trim_a = list(a)
        linreg = st.linregress(trim_n0, trim_a)
        print(linreg)
    
        fig = plt.figure(figsize=(5,5))
        plt.scatter(trim_n0, trim_a, marker='*', color='royalblue')
        #plt.plot([min(trim_n0), max(trim_n0)], 
        #            [min(trim_n0)*linreg.slope + linreg.intercept, 
        #            max(trim_n0)*linreg.slope + linreg.intercept], color='grey')
        plt.xlabel('mu')
        plt.ylabel("beta")
        
        for x in range(len(trim_n0)):
            plt.annotate(x+2, (trim_n0[x]+0.00001, trim_a[x]+0.00001))
            
        plt.hlines(xmin=min(trim_n0)-0.0001, xmax=max(trim_n0)+0.0001, y=0.02, linestyles='--')
    
        if outname != "":
            fig.savefig(f'./graphs/{outname}.png', bbox_inches='tight', dpi=400)
    
    
    
    
    
    
    
    
    #parameter estimation
    
    
    #linear regression or selection of last beta for MM
    
    
    
    
    #qunatile optimization (for LR) and returning best estimates
    
    
    
