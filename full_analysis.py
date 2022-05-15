import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from scipy.optimize import fsolve
import scipy as sc
from pyteomics import pepxml
import random
from collections import deque
import pickle
import importlib as imp
from KDEpy import FFTKDE
from sklearn.metrics import auc
import lower as low
imp.reload(low)

lows = low.Tools()
ems = low.EM()

class Analyze:
    
    def __init__(self, outname):
        self.lower_estimates = []
        self.out = outname
    
    
    def execute_estimation(self, files_paths):
        
        #1. parse the data from pepxml 
        
        tev, charges, big_n = self.parse_pepxmls(files_paths)

        tevs = []
        big_ns = []
        mle_params = []
        mm_params = []
        
        #in this study only 2+, 3+, 4+ spectra are analyzed
        charge_list = [2,3,4]

        for charge in charge_list:
            
            ct, cn  = self.filter_charge(tev, charges, big_n, charge)
            print(f"length of {charge}+ is: {len(ct[:,0])}")
            
            tevs.append(ct)
            big_ns.append(cn)
            
            mle_p = tuple(self.get_mle_params(ct))
            mle_params.append(mle_p)
            
            mm_p = tuple(self.get_mm_params(ct))
            mm_params.append(mm_p)
            
        #get the estimated parameters of top null models for each charge and plot the results
        self.plot_orders(mle_params, mm_params)
        self.plot_mubeta(mle_params, mm_params)
        self.lower_estimates = self.plot_top_models(tevs, mle_params, mm_params)
        #self.lower_estimates = self.alternative_top_models(tevs, mle_params, mm_params)
        
        #if necessary, plot lower order models data, select the charge
        for charge in charge_list:
            self.plot_lower_orders(tevs, mle_params, mm_params, charge)
        
        return self.lower_estimates
    
    @staticmethod
    def get_modes(params):
        linreg = st.linregress(params[0][4:], params[1][4:])
        mean_beta = np.mean(params[1][-3:])
        
        return linreg, mean_beta
    
    @staticmethod
    def get_bic(data, k, order, params):
        data = data[data < 0.17]
        log_params = np.log(params)
        log_like = lows.log_like_mubeta(log_params, data, order)
        bic = k*np.log(len(data)) - 2*log_like
        return bic
    
    @staticmethod
    def get_difference(data, order, params):
        best_mu, best_beta = lows.mle_new(data, order)
        #mu_diff = abs(best_mu-params[0])/best_mu
        beta_diff = abs(best_beta - params[1])/best_beta
        #print(params)
        if params[1] < 0:
            beta_diff = 10e6
        
        #print(f'beta_diff: {beta_diff}')
              
        return beta_diff
    
    
    def get_params_to_compare(self, data, linreg, beta):
        lr_mu, lr_beta = self.qq_lr(data, linreg)
        mean_mu, mean_beta = self.qq_mean(data, beta)
        print(lr_mu, lr_beta, mean_mu, mean_beta)
        
        return lr_mu, lr_beta, mean_mu, mean_beta
    
    
    def alternative_top_models(self, tevs, mle_params, mm_params):
        
        fig, ax = plt.subplots(1,3,figsize=(6, 2))
        params = np.zeros((10,2))
        
        for order in range(3):
            top_hit = tevs[order][:,0]
            #for the purpose of model selection using BIC
            fifth_hit = tevs[order][:,3]
            
            #model selection
            mle_lr, mle_meanbeta = self.get_modes(mle_params[order])
            mm_lr, mm_meanbeta = self.get_modes(mm_params[order])
            modes = [[mle_lr, mle_meanbeta], [mm_lr, mm_meanbeta]]
            bics = []
            alt_params = np.zeros((4,2))     
            
            k=0
            for mode in modes:  
                linreg = mode[0]
                beta = mode[1]
                lr_mu, lr_beta, m_mu, m_beta = self.get_params_to_compare(fifth_hit, linreg, beta)
                
                #k=2 number of parameters estimated, 4 is the index of fifth_hit
                bic_lr = self.get_bic(fifth_hit, 2, 4, [lr_mu, lr_beta])
                bic_m = self.get_bic(fifth_hit, 2, 4, [m_mu, m_beta])
                #bic_lr = self.get_difference(fifth_hit, 3, [lr_mu, lr_beta])
                #bic_m = self.get_difference(fifth_hit, 3, [m_mu, m_beta])
                #print(bic_lr, bic_m)
                alt_params[k,:] = lr_mu, lr_beta
                k += 1
                alt_params[k,:] = m_mu, m_beta
                bics.append(bic_lr)
                bics.append(bic_m)
                k += 1
            
            best_index = bics.index(min(bics))
            print(best_index)
            if best_index == 0:
                best_name = "MLE+LR"
            elif best_index == 1:
                best_name = "MLE+MB"
            elif best_index == 2:
                best_name = "MM+LR"
            elif best_index == 3:
                best_name = "MM+MB"
            
            best_mu, best_beta = alt_params[best_index]
                         
            best_pi = self.find_pi(ax[order], top_hit, best_mu, best_beta)
            params[order+2,:] = [best_mu, best_beta]
            print(best_mu, best_beta, best_pi, best_name)
            
        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_alt_top_models.png", dpi=600, bbox_inches="tight")
        return params
    

    def execute_validation(self, pepxml_file, ref_dict):
        
        #read the pepxml, automatically add the p-values based on lower order estimates
        data = self.validation_df_random(pepxml_file, self.lower_estimates)
        
        #generate decoy-based and EM-based parameters
        decoy_params = self.get_decoy_params(data) 
          
        #get pure EM params      
        _, em_params_em = self.get_em_params(data, outname='em')
        
        #get decoy EM params     
        #_, em_params_dec = self.get_em_params(data, decoy_params, outname='dec')
        
        #get lower EM params     
        #_, em_params_low = self.get_em_params(data, self.lower_estimates, outname='lower')
        
        #add other p-values to the main dataframe
        data = self.add_pvs(data, decoy_params, colname='pv_dec')
        data = self.add_pvs(data, em_params_em, colname='pv_em')
        
        #data = self.add_peps(data, em_params_low, colname='pep_low')
        #data = self.add_peps(data, em_params_dec, colname='pep_dec')
        #data = self.add_peps(data, em_params_em, colname='pep_em')
        #return data
        #"""
        
        #conduct empirical bootstrap on all charges
        
        charges = [2,3,4]
        reps = 200
        
        all_boot_stats = []
        
        for charge in charges:
            
            stats_low = self.bootstrap_stats(data, charge, reps, pv_type='pv_low', mode='single')
            stats_dec = self.bootstrap_stats(data, charge, reps, pv_type='pv_dec', mode='single')
            stats_em = self.bootstrap_stats(data, charge, reps, pv_type='pv_em', mode='single')
            
            all_boot_stats.append([stats_low, stats_dec, stats_em])
            
        fig, ax = plt.subplots(3, 2, figsize=(6,9))
        self.plot_bootstrap_stats(ax[:,0], all_boot_stats)
        self.plot_bootstrap_tps(ax[:,1], all_boot_stats)
        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_validation.png", dpi=600, bbox_inches='tight')
        
        return all_boot_stats
        #"""
    
    #calculate and return the consolidated stats    
    def bootstrap_stats(self, data, charge, reps, pv_type='pv_low', mode='single'):
      
        bootstrap_data = self.bootstrap_fdr(data, charge, reps, colname=pv_type, mode=mode)
        #CI type: 68%
        stats = self.val_stats(bootstrap_data, 0.32)
        return stats
    
    #get statistics of FDP and TP for the investigated FDR values
    def val_stats(self, data, alpha):
        
        length = len(data[0][0,:])

        fdp_stats = np.zeros((3, length))
        tp_stats = np.zeros((3, length))
        fdr_stats = np.zeros((3, length))
        
        fdrs, fdps, tps = data
        
        for i in range(length):
            fdp_stats[:,i] = self.get_cis(fdps, i, alpha)
            tp_stats[:,i] = self.get_cis(tps, i, alpha)
            fdr_stats[:,i] = self.get_cis(fdrs, i, alpha)
        
        return fdr_stats, fdp_stats, tp_stats
    
    #obtain confidence intervals from the empirical bootstrap method
    @staticmethod
    def get_cis(data, idx, alpha):
        
        master_mean = np.mean(data[:,idx])
        diff = sorted([el - master_mean for el in data[:,idx]])
        ci_u = master_mean - diff[int(len(diff)*alpha/2)]
        ci_l = master_mean - diff[int(len(diff)*(1- alpha/2))]
        
        return master_mean, ci_l, ci_u
    
  
     
    def plot_bootstrap_stats(self, ax, all_stats):
        
        #fig, ax = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
        cs = ['royalblue', 'orange', 'green']
        
        for ch in range(3):
            for method in range(3):
                fdrs = all_stats[ch][method][0][0,:]
                fdps = all_stats[ch][method][1]
                if method == 0:
                    self.plot_stats(ax[ch], fdrs, fdps, cs[method], xy=1)
                else:
                    self.plot_stats(ax[ch], fdrs, fdps, cs[method])

            if ch == 2:
                ax[ch].set_xlabel("FDR")
            ax[ch].set_ylabel("FDP")
                           
        #fig.tight_layout()
        #fig.savefig(f"./graphs/{self.out}_fdr_fdp.png", dpi=600, bbox_inches='tight')
        
    def plot_bootstrap_tps(self, ax, all_stats):
        
        #fig, ax = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
        cs = ['royalblue', 'orange', 'green']
        
        for ch in range(3):
            for method in range(3):
                fdrs = all_stats[ch][method][0][0,:]
                tps = all_stats[ch][method][2]
                if method == 0:
                    self.plot_stats(ax[ch], fdrs, tps, cs[method], axis_t="TPR")
                else:
                    self.plot_stats(ax[ch], fdrs, tps, cs[method], axis_t="TPR")

            if ch == 2:
                ax[ch].set_xlabel("FDR")
            ax[ch].set_ylabel("TPR")
                           
        #fig.tight_layout()
        #fig.savefig(f"./graphs/{self.out}_fdr_tpr.png", dpi=600, bbox_inches='tight')
        
        
    #plot the FDP vs FDR results of validation
    @staticmethod
    def plot_stats(ax, fdrs, fdp_stats, col, xy=False, axis_t='FDP'):
        
        #fdrs = np.linspace(0.0001, 0.1, 100)
        
        if xy: ax.plot([0.0001,0.1], [0.0001, 0.1], c='gray')
        
        print(fdp_stats)
        
        ax.plot(fdrs, fdp_stats[0,:], color=col, linewidth=2)
        ax.fill_between(fdrs, fdp_stats[0,:], fdp_stats[2,:], alpha=0.2, color=col)
        ax.fill_between(fdrs, fdp_stats[0,:], fdp_stats[1,:], alpha=0.2, color=col)
        
        #ax.plot(fdrs, fdp_stats[2,:], alpha=0.5, color=col, linestyle='-', linewidth=1)
        #ax.plot(fdrs, fdp_stats[1,:], alpha=0.5, color=col, linestyle='-', linewidth=1)
        
        if axis_t == 'TPR':
            ax.set_xlim(-0.001, 0.1)
            ax.set_ylim(-0.01,)
            
        else:        
            ax.set_xlim(-0.001, 0.1+0.001)
            ax.set_ylim(-0.001, 0.1+0.001)
            
        #ax.set_xlabel("FDR")
        #ax.set_ylabel(axis_t)




    @staticmethod
    def get_decoy_params(df):
        
        charges = np.arange(7) + 1
        params = np.zeros((10,2))
        
        for ch in charges:
            cur_tev = df[(df.charge == ch) & (df.label == 4)]['tev'].to_numpy()
            if len(cur_tev) > 0:
                params[ch, :] = lows.mle_new(cur_tev, 0)
 
        return params
    
    
    
    def get_em_params(self, df, fixed_pars=[], outname="em"):
        
        charges = np.arange(7) + 1
        stats = np.zeros((10,5))
        null_params = np.zeros((10,2))
        
        charges = [2,3,4]
        fig, ax = plt.subplots(1, 3, figsize=(6, 2))
        
        for idx in range(3):
            ch = charges[idx]
            cur_tevs = df[(df.charge == ch) & (df.label != 4)]['tev'].to_numpy()
            if fixed_pars == []:
                params_em = ems.em_algorithm(cur_tevs)
            else:
                params_em = ems.em_algorithm(cur_tevs, fixed_pars[ch])
                
            ems.plot_em(ax[idx], cur_tevs, params_em)
            stats[ch,:] = params_em
            null_params[ch,:] = params_em[:2]
            
        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_EM_{outname}.png", dpi=600, bbox_inches='tight')
    
        return null_params, stats
                


    #find pi0 estimate for the sake of plotting the final fits
    @staticmethod
    def find_pi(ax, data, mu, beta, plot=True):
        axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
        kde = kde/auc(axes, kde)
        trunk = len(axes[axes < 0.2])
        axes_t = axes[:trunk]
        kde_t = kde[:trunk]
        theory = lows.pdf_mubeta(axes, mu, beta, 0)
        theory_t = theory[:trunk]
        error = 1000
        best_pi = 0
        
        for pi in np.linspace(0, 1, 500):
            new_error = abs(auc(axes_t, kde_t) - auc(axes_t, pi*theory_t))
            if new_error < error:
                best_pi = pi
                error = new_error
                
            
        if plot:
            
            ax.fill_between(axes, kde, alpha=0.2, color='green')
            ax.plot(axes, kde, color='green')
            ax.plot(axes, best_pi*theory, color='red', linestyle='-')
            
            ax.set_xlim(0.0, 0.6)
            ax.set_ylim(0,)
            ax.set_xlabel("TEV")
            ax.set_ylabel("density")

        return best_pi

            
        
        

    def plot_top_models(self, tevs, mle_params, mm_params):
        
        def shift(arr, idx):
            return np.sign(arr[idx] - arr[idx+1])
        
        fig, ax = plt.subplots(1,3,figsize=(6, 2), constrained_layout=True)
        params = np.zeros((10,2))
        
        
        for order in range(3):
            top_hit = tevs[order][:,0]
            if mle_params[order][2].rvalue > 0.99 and np.mean(list(map(lambda x: shift(mle_params[order][0], x), range(9)))) < 0:
                print(f"{order}, 'MLE'")
                best_mu, best_beta = self.qq_lr(top_hit, mle_params[order][2])

                if (best_mu < 0) or (best_beta < 0):
                    best_mu, best_beta = self.qq_mean(top_hit, np.mean(mm_params[order][1][-3:]))

            else:
                """
                mm_lr = st.linregress(mm_params[order][0][3:], mm_params[order][1][3:])
                print(mm_lr)
                if abs(mm_lr.rvalue) >= 0.99:
                    print("MM LR")
                    best_mu, best_beta = qq_lr(top_hit, mm_lr)
                if abs(mm_lr.rvalue) < 0.99:"""
                print(f"{order}, 'MM'")
                best_mu, best_beta = self.qq_mean(top_hit, np.mean(mm_params[order][1][-3:]))
                    
            best_pi = self.find_pi(ax[order], top_hit, best_mu, best_beta)
            params[order+2,:] = [best_mu, best_beta]
            print(best_mu, best_beta, best_pi)
            
        #fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_top_models.png", dpi=600, bbox_inches="tight")
        return params
    
    
    


  
    def plot_lower_orders(self, tevs, mle_params, mm_params, idx):
        
        fig, ax = plt.subplots(3,3, figsize=(6, 6), constrained_layout=True)
        sss =1
        charge = idx-2
        
        for row in range(3):
            for col in range(3):
                axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(tevs[charge][:,sss]).evaluate(2**8)
                ax[row%3, col].plot(axes, kde)
                
                mle_kde = lows.pdf_mubeta(axes, mle_params[charge][0][sss], mle_params[charge][1][sss], sss)
                ax[row%3, col].plot(axes, mle_kde)
                
                mm_kde = lows.pdf_mubeta(axes, mm_params[charge][0][sss], mm_params[charge][1][sss], sss)
                ax[row%3, col].plot(axes, mm_kde)
                
                ax[row%3, col].set_ylim(0,)
                
                if col == 0:
                    ax[row%3, col].set_ylabel("density")
                if row == 2:
                    ax[row%3, col].set_xlabel("TEV")
                    
                #ax[row%3, col].set_xlim(0, 0.4)
                sss += 1
                
        #fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_lower_models_{idx}.png", dpi=600, bbox_inches="tight")
        
        
    @staticmethod    
    def scatter_params(params, outname="example"):
        
        x=3
        fig, ax = plt.subplots(figsize=(4,4))

        for i in range(len(params)):
            ax.scatter(params[i][0][x:], params[i][1][x:])
        
        ax.set_xlabel("mu")
        ax.set_ylabel("beta")
        ax.set_title("testing")
        ax.legend(['2+', '3+', '4+'])
        #fig.savefig(f'{outname}_params_scatter.png', dpi=400, bbox_inches='tight')
        

    def plot_lower_hist(self, tev, params, alpha):
        fig, ax = plt.subplots(3,3, figsize=(4,4))
        sss =1
        for row in range(3):
            for col in range(3):
                self.plot_fit(ax[row%3, col], tev[alpha][:,sss], params[alpha][0][sss], params[alpha][1][sss], sss, col='blue', frac=1, bins=500)
                sss += 1
        #fig.savefig('yeast_3Da_1Da_f_lowerhits.png', dpi=400, bbox_inches='tight')
        
        
      
    def plot_orders(self, mle_params, mm_params):
        no_orders = 10
        fig, ax = plt.subplots(2,3, figsize=(6,3), constrained_layout=True)
    
        for row in range(2):
            for col in range(3):

                    ax[row, col].scatter(np.arange(no_orders)+1, mle_params[col][row], marker='.')
                    ax[row, col].scatter(np.arange(no_orders)+1, mm_params[col][row], marker='.')
                    ax[row, col].set_xticks(np.arange(10)+1)
                    if row == 0:
                        ax[row, col].set_ylabel(r"$\mu$")
                    else:
                        ax[row, col].set_ylabel(r"$\beta$")
                    
                    if row == 1:
                        ax[row, col].set_xlabel("order")
        
        #fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mle_mm_params.png", dpi=600, bbox_inches="tight")

    
    def plot_mubeta(self, mle_params, mm_params):
        no_orders = 10
        fig, ax = plt.subplots(1,3, figsize=(9,3))
    
        for row in range(3):
            mle_c = 'royalblue'
            mm_c = 'orange'
            mle_x, mle_y = mle_params[row][0][3:], mle_params[row][1][3:]
            mm_x, mm_y = mm_params[row][0][3:], mm_params[row][1][3:]

            ax[row].scatter(mle_x, mle_y, color=mle_c, marker='o', edgecolors='k',linewidths=0.5)
            ax[row].scatter(mm_x, mm_y, color=mm_c, marker='o', edgecolors='k',linewidths=0.5)

            print(f"charge {row+2}, MLE params")
            self.annotation(ax[row], mle_x, mle_y, mle_c)
            print(f"charge {row+2}, MM params")
            self.annotation(ax[row], mm_x, mm_y, mm_c)

            ax[row].set_xlabel(r"$\mu$")
            if row == 0:
                ax[row].set_ylabel(r"$\beta$")
              
        
        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mubeta_params_numbered.png", dpi=600, bbox_inches="tight")

        fig, ax = plt.subplots(1,3, figsize=(9,3))
    
        for row in range(3):
            mle_c = 'royalblue'
            mm_c = 'orange'
            mle_x, mle_y = mle_params[row][0][3:], mle_params[row][1][3:]
            mm_x, mm_y = mm_params[row][0][3:], mm_params[row][1][3:]

            ax[row].scatter(mle_x, mle_y, color=mle_c, marker='o', edgecolors='k',linewidths=0.5)
            ax[row].scatter(mm_x, mm_y, color=mm_c, marker='o', edgecolors='k',linewidths=0.5)

            #self.annotation(ax[row], mle_x, mle_y, mle_c)
            #self.annotation(ax[row], mm_x, mm_y, mm_c)

            ax[row].set_xlabel(r"$\mu$")
            if row == 0:
                ax[row].set_ylabel(r"$\beta$")

        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mubeta_params_clean.png", dpi=600, bbox_inches="tight")

    
    def annotation(self, ax, x, y, col):

        offset = 3
        for item in range(len(x)):
            ax.annotate(item+offset, (x[item], y[item]-0.0002), color=col)

        #plot linreg
        linreg = st.linregress(x, y)
        print(linreg.rvalue)
        #xs = np.array([min(x), max(x)])
        #ymin, ymax = linreg.slope*xs + linreg.intercept
        #ax.plot(xs, [ymin, ymax], color=col)
        #ax.set_title(f"{linreg.rvalue}_{linreg.pvalue}")

        
        
    def plot_mu_beta_lr(self, mle_params, mm_params):
        
        fig, ax = plt.subplots(1,3, figsize=(6, 2))
        offset = 2
        
        for col in range(3):
            
            ax[col].scatter(mle_params[col][0][offset:], mle_params[col][1][offset:], marker='.', color='royalblue')
            ax[col].scatter(mm_params[col][0][offset:], mm_params[col][1][offset:], marker='.', color='orange')
            
            if col == 0:
                ax[col].set_ylabel(r"$\beta$")
                ax[col].set_xlabel(r"$\mu$")
            else:
                ax[col].set_xlabel(r"$\mu$")
        
        
        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mubeta_LR.png", dpi=600, bbox_inches="tight")
    
        
        
    #input: pepxml files, output: np arrays (TEV score, charge, N)
    @staticmethod
    def parse_pepxmls(paths):

        tev = deque()
        charges = deque()
        big_n  = deque()

        for path in paths:
            print(path)
            cur_file = pepxml.read(path)

            for spec in cur_file:
                if 'search_hit' in spec.keys():
                    if len(spec['search_hit']) == 10:
                        tev.append(list(map(lambda x: -0.02*np.log(spec['search_hit'][x]['search_score']['expect']/1000), np.arange(10))))
                        charges.append(int(spec['assumed_charge']))
                        big_n.append(int(spec['search_hit'][0]['num_matched_peptides']))

        return tev, charges, big_n

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
        
        mus = []
        betas = []

        for hit in range(10):
            #if hit == 0: continue
            #print(hit)
            cur_tev = arr[:,hit].astype('float128')
            cur_tev = cur_tev[cur_tev > 0]
            cur_tev = sorted(cur_tev)
            #length = len(cur_tev)
            #cur_tev = cur_tev[int(length*0.05):int(length*0.95)]
            cur_mu, cur_beta = lows.mle_new(cur_tev, hit)
            mus.append(cur_mu)
            betas.append(cur_beta)
            
        return mus, betas
    
    """
    @staticmethod
    def plot_fitted( arr, N0, a, alpha, col='blue', frac=1, bins=500):
        sorted_arr = np.array(sorted(arr))
        l_lim = sorted_arr[0]
        u_lim = sorted_arr[-1]
        pdf = lows.pdf_mubeta(sorted_arr, N0, a, alpha)
        plt.plot(sorted_arr, frac*pdf,color=col)
        sns.distplot(sorted_arr, bins = np.linspace(0, 0.8, bins), kde=False, norm_hist=True,
                    hist_kws=dict(histtype='step', linewidth=1, color='black'))
        #ax.set_xlim(l_lim, u_lim)
        plt.xlim(l_lim, u_lim)
        """
        
        
    @staticmethod
    def plot_fit(ax, arr, N0, a, alpha, col='blue', frac=1, bins=500):
        sorted_arr = np.array(sorted(arr))
        #l_lim = sorted_arr[0]
        #u_lim = sorted_arr[-1]
        pdf = lows.pdf_mubeta(sorted_arr, N0, a, alpha)
        ax.plot(sorted_arr, frac*pdf,color=col)
        ax.hist(sorted_arr, bins = np.linspace(0, 0.8, bins), histtype='step', density=True)
        median = np.median(sorted_arr)
        ax.vlines(x=median, ymin=0, ymax=20)
        #ax.set_xlim(l_lim, u_lim)
        ax.set_xlim(0, 0.3)


    def plot_params(self, n0, a, xxx=0):

        trim_n0 = list(n0)
        trim_a = list(a)
        linreg = st.linregress(trim_n0, trim_a)
        print(linreg)

        fig = plt.figure(figsize=(4,4))
        plt.scatter(trim_n0, trim_a, marker='o', color='royalblue')
        plt.plot([min(trim_n0), max(trim_n0)], 
                    [min(trim_n0)*linreg.slope + linreg.intercept, 
                    max(trim_n0)*linreg.slope + linreg.intercept], color='grey')
        plt.xlabel('mu')
        plt.ylabel("beta")
        plt.xlim(min(trim_n0)-0.001, max(trim_n0)+0.001)
        
        for x in range(len(trim_n0)):
            plt.annotate(x+xxx, (trim_n0[x]+0.00001, trim_a[x]+0.00003))
            
        #plt.hlines(xmin=min(trim_n0)-0.0001, xmax=max(trim_n0)+0.0001, y=0.02, linestyles='--')

        fig.savefig(f'./graphs/{self.out}_params.png', bbox_inches='tight', dpi=600)
            
            
############### PARAMETER ESTIMATION ###################################            
            
            
    #obtain MLE parameters and their fitted linear regression mu vs. beta
    def get_mle_params(self, tevs, cutoff=3):
        
        n0, a = self.lower_params(tevs)
        trim_n0 = list(n0)[cutoff:]
        trim_a = list(a)[cutoff:]
        linreg = st.linregress(trim_n0, trim_a)

        return n0, a, linreg

    #obtain method of moments parameters
    @staticmethod
    def get_mm_params(tev):
        m1 = []
        m2 = []
        for order in range(10):
            cur_m1, cur_m2 = lows.mm_estimator(tev[:,order], order)
            m1.append(cur_m1)
            m2.append(cur_m2)
        return m1, m2


###########################################################


################# QUANTILE OPTIMIZATION ###################


    def qq_mean(self, tev, opt_beta):
        
        emps = sorted(tev)
        emps_df = pd.DataFrame(emps)

        emps_df[1] = emps_df.index + 1
        #empirical cdf
        emps_df[2] = emps_df[1]/len(emps)
    
        emp_cdf = emps_df[2].to_numpy()[:-1]
        emp_q = emps_df[0].to_numpy()[:-1]

        errors = []
        qq_range = np.linspace(0.05, 0.4, 500)

        for i in qq_range:
          
            """
            theor_q = lows.gumbel_new_ppf(emp_cdf, i, opt_beta)
            diffs = abs(np.subtract(theor_q, emp_q))/emp_q

            length = len(emp_q[emp_q < 0.15])
            
            diffs = diffs[:length]
            diffs = np.mean(diffs)
            """
            
            k=1
            diffs = self.get_bic(tev, k, 0, [i, opt_beta])
            
            errors.append(diffs)

        opt_idx = errors.index(min(errors))
        opt_mu = qq_range[opt_idx]
        
        fig = plt.figure(figsize=(4,4))
        plt.plot(qq_range, errors)
        plt.scatter(qq_range, errors, s=1)
        #plt.xlim(60,70)
        plt.xlabel("N0")
        plt.ylabel("loss")
        #plt.savefig(f"./graphs/{self.out}_optim_loss.png", dpi=600, bbox_inches='tight')

        theor_q = lows.gumbel_new_ppf(emp_cdf, opt_mu, opt_beta)

        fig1 = plt.figure(figsize=(4,4))
        plt.plot([0, 1], [0, 1], color='k')
        plt.scatter(emp_q, theor_q, s=3)

        plt.xlim(0, 0.2)
        plt.ylim(0, 0.2)
        plt.xlabel("theoretical quantile")
        plt.ylabel('empirical quantile')
        #plt.savefig(f"./graphs/{self.out}_optim_QQ.png", dpi=600, bbox_inches='tight')
        
        return opt_mu, opt_beta

  
    # generate quantiles
    def qq_lr(self, tev, linreg):

        emps = sorted(tev)
        emps_df = pd.DataFrame(emps)
        emps_df[1] = emps_df.index + 1
        
        #empirical cdf
        emps_df[2] = emps_df[1]/len(emps)
        emp_cdf = emps_df[2].to_numpy()[:-1]
        emp_q = emps_df[0].to_numpy()[:-1]

        #quantile optimization
        errors = []
        qq_range = np.linspace(0.05, 0.4, 500)
        #opt_a = np.mean(mms[:,1][-3:])
        for i in qq_range:
            cur_a = i*linreg.slope + linreg.intercept
            """
            #cur_a = opt_a
            theor_q = lows.gumbel_new_ppf(emp_cdf, i, cur_a)
            diffs = abs(np.subtract(theor_q, emp_q))/emp_q

            length = len(emp_q[emp_q < 0.15])
            
            diffs = diffs[:length]
            diffs = np.mean(diffs)
            """
            
            k=1
            diffs = self.get_bic(tev, k, 0, [i, cur_a])
            errors.append(diffs)

        opt_idx = errors.index(min(errors))
        opt_N0 = qq_range[opt_idx]
        opt_a = opt_N0*linreg.slope + linreg.intercept
        
        fig = plt.figure(figsize=(4,4))
        plt.plot(qq_range, errors)
        plt.scatter(qq_range, errors, s=1)
        #plt.xlim(60,70)
        plt.xlabel("N0")
        plt.ylabel("loss")
        #plt.savefig(f"./graphs/{self.out}_optim_loss.png", dpi=600, bbox_inches='tight')

        theor_q = lows.gumbel_new_ppf(emp_cdf, opt_N0, opt_a)
        
        fig1 = plt.figure(figsize=(4,4))
        plt.plot([0, 1], [0, 1], color='k')
        plt.scatter(emp_q, theor_q, s=3)

        plt.xlim(0, 0.2)
        plt.ylim(0, 0.2)
        plt.xlabel("theoretical quantile")
        plt.ylabel('empirical quantile')
        #plt.savefig(f"./graphs/{self.out}_optim_QQ.png", dpi=600, bbox_inches='tight')
        
    
        return opt_N0, opt_a
    
    ###########################################################
    
    
   ############### VALIDATION #################################
   
   
    @staticmethod
    def validation_df_random(paths, pars):
        
        length = 500000
        pvs = -1*np.ones(length)
        labels = np.zeros(length)
        charges = np.zeros(length)
        tevss = np.zeros(length)
        k=0
        
        for pepxml_file in paths:
            d = pepxml.read(pepxml_file)
            
            for el in d:
                if 'search_hit' in el.keys():

                    tev = -0.02 * np.log((el['search_hit'][0]['search_score']['expect']) / 1000)
                    ch = int(el['assumed_charge'])
                    spec = el['spectrum']
                    
                    if ch not in [2,3,4]:
                        p_v = 1
                    else:
                        p_v = 1 - lows.mubeta_cdf(tev, pars[ch][0], pars[ch][1])

                    label = 3
                    if "decoy" in spec:
                        label = 4
                    elif "random" in spec:
                        label = 0
                    elif "pos" in spec:
                        label = 1
  
                    pvs[k] = p_v
                    labels[k] = label
                    charges[k] = ch
                    tevss[k] = tev
                    k += 1
                
        df = pd.DataFrame(np.array([pvs, labels, charges, tevss]).T)
        df.columns = ['pv_low', 'label', 'charge', 'tev']
        df = df[df['pv_low'] != -1]       
        
        return df

    #parse pepxml and ground truth info of the validation dataset
    @staticmethod
    def validation_df(pepxml_file, ref_dict, params):
        
        d = pepxml.read(pepxml_file)
        ref_p = pickle.load(open(ref_dict, "rb"))
        
        #IDs = np.zeros(len(d))
        pvs = -1*np.ones(len(d))
        labels = np.zeros(len(d))
        charges = np.zeros(len(d))
        tevss = np.zeros(len(d))
        k=0
        for el in d:
            
            if 'search_hit' in el.keys():

                scanid = int(el['start_scan'])
                tev = -0.02 * np.log((el['search_hit'][0]['search_score']['expect']) / 1000)
                charge = int(el['assumed_charge'])
               
                if params[charge][0] == 0:
                    p_v = 1
                else:
                    p_v = 1 - lows.mubeta_cdf(tev, params[charge][0], params[charge][1])

                pep = el['search_hit'][0]['peptide']
                new_seq = pep.replace('I', 'X').replace('L', 'X')
                label = 0
                
                
                if scanid in ref_p.keys():
                    if ref_p[scanid] == new_seq:
                        label = 1
                    else:
                        label = 0
                if scanid not in ref_p.keys():
                    label = 4
                    
                if "random" in el['spectrum']:
                    label = 3
                    
                    

                #if 'DECOY' in el['search_hit'][0]['proteins'][0]['protein']:
                #    label = 4
                        
                pvs[k] = p_v
                labels[k] = label
                charges[k] = charge
                tevss[k] = tev
                k += 1
                
        df = pd.DataFrame(np.array([pvs, labels, charges, tevss]).T)
        df.columns = ['pv_low', 'label', 'charge', 'tev']
        df = df[df['pv_low'] != -1]       
        
        return df
    
    
    @staticmethod
    def BH(df, bh, pv_name='pv_low'):
        
        df.loc[:,'bh'] = bh.values
        finaldf = df[df[pv_name] <= df['bh']]
        
        return finaldf


    #generate bootstrapped fdp estimates
    #get subsample from df and calculate the stats for it, accumulate the stats 
    def bootstrap_fdr(self, df, ch, reps, colname='pv_low', mode='single'):
        
        work_df = df[(df["charge"] == ch) & (df.label != 4)]
        length = np.arange(len(work_df))
        
        #fdrs = np.linspace(0.0001, 0.1, 100)
        fdrs = np.zeros((reps, 100))
        fdps = np.zeros((reps, 100))
        tps = np.zeros((reps,100))
        
        l = len(work_df)
        em_fdps = np.zeros((reps, l))
        em_tps = np.zeros((reps, l))
        em_fdrs = np.zeros((reps, l))
        
        for rep in range(reps):
           
            random.seed()
            new_sel = random.choices(length, k=len(length))
            new_df = work_df.iloc[new_sel, :]
            if mode == 'single':
                fdr, fdp, tp = self.fdr_lower(new_df, ch, colname=colname)
                fdps[rep, :] = fdp
                tps[rep,:] = tp
                fdrs[rep,:] = fdr
            else:
                fdr, fdp, tp = self.pep_fdr(new_df, ch, colname=colname)
                em_fdps[rep, :] = fdp
                em_tps[rep,:] = tp
                em_fdrs[rep,:] = fdr
                
        if mode == 'single':
            return fdrs, fdps, tps
        else:
            return em_fdrs, em_fdps, em_tps
    
    #add alternative p-values to the main dataframe
    @staticmethod
    def add_pvs(df, params, colname='pv_em'):
        
        pvs = np.zeros(len(df))
        
        for pos, idx in enumerate(df.index):
            cur_tev = df.loc[idx, 'tev']
            ch = int(df.loc[idx, 'charge'])
            pv = 1 - lows.mubeta_cdf(cur_tev, params[ch][0], params[ch][1])
            pvs[pos] = pv
   
        df[colname] = pvs
        
        return df
    
    @staticmethod
    def add_peps(df, params, colname='pep_em'):
        
        pvs = np.zeros(len(df))
    
        for pos, idx in enumerate(df.index):
            cur_tev = df.loc[idx, 'tev']
            ch = int(df.loc[idx, 'charge'])
            old_mu1, old_beta, old_mu2, old_sigma, old_pi0 = params[ch]
            
            if ch in [2,3,4]:
                neg = lows.pdf_mubeta(cur_tev, old_mu1, old_beta, 0)
                posit = st.norm.pdf(cur_tev, old_mu2, old_sigma)
                pep = old_pi0*neg/(old_pi0*neg + (1-old_pi0)*posit)
            else:
                pep = 1
                
            #to prevent inaccurate fit of positive model to mess up the results
            if cur_tev <=0.15:
                pep = 1
            
            pvs[pos] = pep
   
        df[colname] = pvs
        
        return df
    
    @staticmethod
    def pep_fdr(df, ch, colname):
            
        #colname is the name of PEP column
        
        df = df[(df["label"] != 4) & (df["charge"] == ch)]
        df.sort_values(colname, ascending=True, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.index += 1
        df['fdr'] = df[colname].cumsum()/df.index
        df['fdp'] = (df.index - df['label'].cumsum())/df.index
        df['tp'] = df['label'].cumsum()/len(df[df['label'] == 1])
        
        return df['fdr'].to_numpy(), df['fdr'].to_numpy(), df['tp'].to_numpy()
        
    
    

    
    
    
   #generate the data of FDP and TP for the selected FDR range
    def fdr_lower(self, df, ch, colname='pv_low'):
        
        fdps = []
        fdrs = np.linspace(0.0001, 0.1, 100)
        #decs = []
        tps = []
        
        #select only target PSMs of the desired charge
        
        df = df[(df["label"] != 4) & (df["charge"] == ch)]
        df.sort_values(colname, ascending=True, inplace=True)
        df.reset_index(inplace=True, drop=True)
        df.index += 1
        

        for fdr in fdrs:
            
            bh = pd.Series((df.index.to_series() * fdr) / len(df))
            adj = self.BH(df, bh, colname)
            len_accepted = len(adj)
            len_correct = len(df[df.label==1])
            
            if len_accepted == 0: len_accepted = 1
            if len_correct == 0: len_correct = 1
            
            len_tps = len(adj[adj['label'] == 1])
            
            fdp = 1-len_tps/len_accepted
            #dec = 2*len(ch3[ch3['label'] == 4])/len(ch3)
            tp = len_tps/len_correct
            
            
            tps.append(tp)
            fdps.append(fdp)
            #decs.append(dec)
            
            
        return fdrs, fdps, tps
            
    #plot both FDP and TP vs FDR in the same plot
    @staticmethod
    def plot_val_results(ax, fdrs, fdps, tps, lim, col1, col2):
        
        #fig, ax = plt.subplots(figsize=(6,6))
        #lim = 0.1
        ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.2)
        ax.plot(fdrs, fdps, color=col1)
        ax.scatter(fdrs, fdps, marker='.', color=col1)
        ax.set_ylabel("FDP")
        ax2=ax.twinx()
        #ax.plot(fdr, decs)
        #ax.scatter(fdr, decs, marker='.')
        ax2.plot(fdrs, tps, marker='.', color=col2)
        ax2.set_ylabel("TPR", color=col2)
        ax2.set_ylim(0,1)
        ax.plot([0,lim], [0,lim], color='k', alpha=0.5)
        ax.set_xlim(-0.001, lim)
        ax.set_ylim(-0.001, lim)
        #ax2.legend(['lower', 'decoys', 'x-y'])
        
        
    #calculation of FDP based on decoy counting
    @staticmethod    
    def fdr_dec(df, ch):
        fdp = []
        fdrs = []
        decs = []
        peps_low = []
        peps_decs = []
        tps = []
        df = df[df.charge==ch]
        df.sort_values("tev", ascending=False, inplace=True)

        for i in np.linspace(1, len(df), 1000):
            if i == 0: continue
            ch3 = df.iloc[:int(i), :]
            c = 1-len(ch3[ch3['label'] == 1])/len(ch3)
            dec = 2*len(ch3[ch3['label'] == 4])/len(ch3)
            tp = len(ch3[ch3.label == 1])/len(df[df.label==1])
            tps.append(tp)
            fdp.append(c)
            fdrs.append(dec)
            
        fig, ax = plt.subplots(figsize=(6,6))
        lim = 0.1
        ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.2)
        ax.plot(fdrs, fdp, color='royalblue')
        ax.scatter(fdrs, fdp, marker='.', color='royalblue')
        ax.set_ylabel("FDP", color='royalblue')
        ax2=ax.twinx()
        #ax.plot(fdr, decs)
        #ax.scatter(fdr, decs, marker='.')
        ax2.plot(fdrs, tps, color='orange')
        ax2.set_ylabel("TPR", color='orange')
        ax2.set_ylim(0,1)
        ax.plot([0,lim], [0,lim], color='k', alpha=0.5)
        ax.set_xlim(-0.001, lim)
        ax.set_ylim(-0.001, lim)
        #ax2.legend(['lower', 'decoys', 'x-y'])
        return fdrs, fdp, tps
    
     ###########################################################
                        