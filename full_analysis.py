"""Full analysis of pepxml file using lower order statistics"""
import random
from collections import deque
import pickle
import importlib as imp
import functools as fu
from xml.etree import ElementTree as ET
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from pyteomics import pepxml
from KDEpy import FFTKDE
from sklearn.metrics import auc
import lower as low
imp.reload(low)
warnings.filterwarnings("ignore")

lows = low.Tools()
ems = low.EM()

class Analyze:
    """
    Analyze the results to get the top model parameters

    ...

    Attributes
    ----------
    

    Methods
    -------

    execute_estimation(self, file_paths, param_outname)
        Executes parameter estimation for top models using
        lower order statistics

    
    
    """

    def __init__(self, outname, top_n=30, bic_cutoff=0.17):
        self.params_est = []
        self.out = outname
        self.len_correct = 0
        self.reps = 500
        self.top_n = top_n
        self.bic_cutoff = bic_cutoff

    def execute_estimation(self, files_paths, param_outname):
        """Exectues parameter estimation for top model using lower
        order statistics

        Parameters
        ----------
        files_paths : list
            List of paths of pep.xml files to use for parameter estimation
        param_outname : str
            Core name of all output files for the estimation process

        Returns
        -------
        tuple
            a tuple of estimated top model parameters, MLE, and MM estimates for lower-order models
        """


        tevs, charges = self.__fast_parse(files_paths)
        tevs = np.nan_to_num(tevs)
        all_charges = [2,3,4]

        #for each charge, extract scores and estimate MLE/MM params
        data = list(map(fu.partial(self.__process_params, tevs, charges), all_charges))
        tevs, mle_pars, mm_pars = list(zip(*data))
        mm_pars = np.nan_to_num(np.array(mm_pars))


        #get the estimated parameters of top null models for each charge and plot the results
        #self.__plot_orders(mle_pars, mm_pars)
        #output = self.plot_mubeta(mle_params, mm_params)
        #self.lower_estimates = self.plot_top_models(tevs, mle_params, mm_params)
        self.params_est = self.__alt_top_models(tevs, mle_pars, mm_pars, len(all_charges))

        for charge in all_charges:
            self.__plot_lower_models(tevs, mle_pars, mm_pars, charge)

        self.__export_pars_to_txt(param_outname)

        return self.params_est, mle_pars, mm_pars, tevs, charges


    def __export_pars_to_txt(self, pars_outname):
        """export params to txt for modified PeptideProphet (mean & std)"""

        params = pd.DataFrame(self.params_est[1:])
        params[0] = params[0] + params[1]*np.euler_gamma
        params[1] = np.pi/np.sqrt(6)*params[1]
        params.to_csv(f"{pars_outname}.txt", sep=" ", header=None, index=None)


    def __process_params(self, tevs, charges, sel_charge):
        """calculate MLE and MM parameters using the data"""
        sel_scores = tevs[np.where((charges == sel_charge))]
        mle_pars = tuple(self.__get_mle_params(sel_scores))
        mm_pars = tuple(self.__get_mm_params(sel_scores))
        return sel_scores, mle_pars, mm_pars

    @staticmethod
    def __get_modes(pars, first_idx=4, last_beta_idx=-3):
        """get linear regression parameters and mean beta for further processing"""
        mu_idx, beta_idx = [0,1]
        linreg = st.linregress(pars[mu_idx][first_idx:],
                                pars[beta_idx][first_idx:]) # drop first 3 points
        mean_beta = np.mean(pars[1][last_beta_idx:]) # use last 3 data points
        return linreg, mean_beta

    def __get_bic(self,data, k, order, params):
        """calculate BIC for lower section of the distribution"""
        data = data[data < self.bic_cutoff]
        log_like = lows.log_like_mubeta(np.log(params), data, order)
        bic = k*np.log(len(data)) - 2*log_like
        return bic

    @staticmethod
    def __get_beta_diff(data, order, params):
        """calculate beta difference between best models and supplied value"""
        _, best_beta = lows.mle_new(data, order)
        #mu_diff = abs(best_mu-params[0])/best_mu
        beta_diff = abs(best_beta - params[1])/best_beta
        #print(params)
        #if params negative ie not determined well, axe the case
        #by assigning big beta difference
        if params[1] < 0:
            beta_diff = 10e6

        return beta_diff

    def __get_alt_params(self, data, estim_mode):
        """generate LR and mean beta params for given data"""
        linreg, beta = estim_mode
        lr_mu, lr_beta = self.__bic_optimization(data, linreg=linreg)
        mean_mu, mean_beta = self.__bic_optimization(data, mean_beta=beta)
        return lr_mu, lr_beta, mean_mu, mean_beta


    def __alt_top_models(self, tevs, mle_params, mm_params, charge_no):
        """estimate parameters of top models using lower order distributions"""
        fig, axes = plt.subplots(1, charge_no, figsize=(2*charge_no, 2))
        params = np.zeros((10,2))

        for ch_idx in range(charge_no):

            top_hit = tevs[ch_idx][:,0]
            mode_params = []
            bics = []
            mode_params.append(self.__get_modes(mle_params[ch_idx]))
            mode_params.append(self.__get_modes(mm_params[ch_idx]))
            tmp_params = []

            for mode in mode_params:
                tmp_params, bics = self.__get_bics_data(top_hit, mode, tmp_params, bics)

            best_idx = bics.index(min(bics))
            _ = self.__find_pi(axes[ch_idx], top_hit, tmp_params[best_idx]) # for plotting
            params[ch_idx+2,:] = tmp_params[best_idx]

        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_alt_top_models.png", dpi=600, bbox_inches="tight")
        return params


    def __get_bics_data(self, top_hit, mode, tmp_params, bics):
        """obtain BICs and temporary parameters"""
        lr_mu, lr_beta, m_mu, m_beta = self.__get_alt_params(top_hit, mode)

        bic_lr = 0.8*self.__get_bic(top_hit, 2, 0, [lr_mu, lr_beta])
        bic_m = self.__get_bic(top_hit, 2, 0, [m_mu, m_beta])

        tmp_params.append([lr_mu, lr_beta])
        tmp_params.append([m_mu, m_beta])

        if (lr_mu < 0) or (lr_beta < 0):
            bic_lr = 1e6

        bics.append(bic_lr)
        bics.append(bic_m)

        return tmp_params, bics






    def execute_validation(self, pepxml_file):
        """read the pepxml, automatically add the p-values based on lower order estimates"""
        #TODO: this needs to be rewritten to validate against Nokoi and Peng

        #data = self.validation_df_random(pepxml_file, self.lower_estimates)
        scores, charges, lower_pvs, labels = self.__faster_validation(pepxml_file, self.params_est)

        decoy_params = self.__get_decoy_params(charges, labels, scores)
        _, em_params_em = self.__get_em_params(charges, labels, scores, outname='em')

        #get decoy EM params
        #_, em_params_dec = self.get_em_params(data, decoy_params, outname='dec')

        #get lower EM params
        #_, em_params_low = self.get_em_params(data, self.lower_estimates, outname='lower')

        decoy_pvs = self.faster_add_pvs(scores, charges, decoy_params)
        em_pvs = self.faster_add_pvs(scores, charges, em_params_em)

        #data = self.add_peps(data, em_params_low, colname='pep_low')
        #data = self.add_peps(data, em_params_dec, colname='pep_dec')
        #data = self.add_peps(data, em_params_em, colname='pep_em')
        #return data

        #conduct empirical bootstrap on all charges

        chars = [2,3,4]

        all_boot_stats = deque()
        idx_nondecoys = set(np.where(labels != 4)[0])

        for charge in chars:
            #prt(f"this is charge {charge}...")

            idx_charges = set(np.where(charges == charge)[0])
            idx_shared = list(set.intersection(idx_nondecoys, idx_charges))
            cur_labels = labels[idx_shared]
            cur_lower = lower_pvs[idx_shared]
            cur_decoy = decoy_pvs[idx_shared]
            cur_em = em_pvs[idx_shared]
            self.len_correct = len(cur_labels[cur_labels == 1])
            
            stats_low = self.__bootstrap_stats(cur_labels, cur_lower)
            stats_dec = self.__bootstrap_stats(cur_labels, cur_decoy)
            stats_em = self.__bootstrap_stats(cur_labels, cur_em)
            
            all_boot_stats.append([stats_low, stats_dec, stats_em])
            
        fig, ax = plt.subplots(3, 2, figsize=(6,6))
        self.__plot_bootstrap_stats(ax[:,0], all_boot_stats)
        self.__plot_bootstrap_tps(ax[:,1], all_boot_stats)
        fig.tight_layout()
        
        fig.savefig(f"./graphs/{self.out}_validation.png", dpi=600, bbox_inches='tight')
        
        return all_boot_stats


    def __bootstrap_stats(self, labels, pvs):
        """calculate the consolidated stats"""
        bootstrap_data = self.__bootstrap_fdr(self.reps, labels, pvs, self.len_correct)
        bootstrap_data = np.array(bootstrap_data)
        stats = self.__val_stats(bootstrap_data, 0.32) # CI type: 68%
        return stats


    def __val_stats(self, data, alpha):
        """get FDP and TP stats for the tested FDR values"""

        length = len(data[0][0,:])
        fdp_stats = np.zeros((3, length))
        tp_stats = np.zeros((3, length))
        fdr_stats = np.zeros((3, length))

        fdrs, fdps, tps = data

        for i in range(length):
            fdp_stats[:,i] = self.__get_cis(fdps, i, alpha)
            tp_stats[:,i] = self.__get_cis(tps, i, alpha)
            fdr_stats[:,i] = self.__get_cis(fdrs, i, alpha)

        return fdr_stats, fdp_stats, tp_stats

    @staticmethod
    def __get_cis(data, idx, alpha):
        """obtain CIs from empirical bootstrap method"""

        master_mean = np.mean(data[:,idx])
        diff = sorted([el - master_mean for el in data[:,idx]])
        ci_u = master_mean - diff[int(len(diff)*alpha/2)]
        ci_l = master_mean - diff[int(len(diff)*(1- alpha/2))]

        return master_mean, ci_l, ci_u

    def __plot_bootstrap_stats(self, ax, all_stats):
        """plotting bootstrap results"""

        #fig, ax = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
        cs = ['#2D58B8', '#D65215', '#2CB199']
        pis = [1.67, 1.1, 1.1]

        for ch in range(3):
            for method in range(3):
                fdrs = all_stats[ch][method][0][0,:]
                fdps = all_stats[ch][method][1]
                if method == 0:
                    self.__plot_fdp_fdr(ax[ch], fdrs, pis[ch]*np.array(fdps), cs[method], xy=1)
                else:
                    self.__plot_fdp_fdr(ax[ch], fdrs, pis[ch]*np.array(fdps), cs[method])

            if ch == 2:
                ax[ch].set_xlabel("FDR")
            ax[ch].set_ylabel("FDP")
                           
        #fig.tight_layout()
        #fig.savefig(f"./graphs/{self.out}_fdr_fdp.png", dpi=600, bbox_inches='tight')
        
    def __plot_bootstrap_tps(self, ax, all_stats):
        
        #fig, ax = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
        cs = ['#2D58B8', '#D65215', '#2CB199']
        
        for ch in range(3):
            for method in range(3):
                fdrs = all_stats[ch][method][0][0,:]
                tps = all_stats[ch][method][2]
                if method == 0:
                    self.__plot_fdp_fdr(ax[ch], fdrs, tps, cs[method], axis_t="TPR")
                else:
                    self.__plot_fdp_fdr(ax[ch], fdrs, tps, cs[method], axis_t="TPR")

            if ch == 2:
                ax[ch].set_xlabel("FDR")
            ax[ch].set_ylabel("TPR")
                           
        #fig.tight_layout()
        #fig.savefig(f"./graphs/{self.out}_fdr_tpr.png", dpi=600, bbox_inches='tight')
        
        
    #plot the FDP vs FDR results of validation
    @staticmethod
    def __plot_fdp_fdr(ax, fdrs, fdp_stats, col, xy=False, axis_t='FDP'):
        """plot FDP vs FDR - results of validation"""
        #fdrs = np.linspace(0.0001, 0.1, 100)

        if xy:
            ax.plot([0.0001,0.1], [0.0001, 0.1], c='gray')
        
        #print(fdp_stats)
        
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
    def __get_decoy_params(charges, labels, scores):
        """obtain parameters from decoy PSMs"""
        ch_idx = np.arange(7) + 1
        params = np.zeros((10,2))
        idx_labels = set(np.where(labels == 4)[0])
                
        for ch in ch_idx:
            
            idx_charges = set(np.where(charges == ch)[0])
            idx_shared = list(set.intersection(idx_charges, idx_labels))
            cur_scores = scores[idx_shared]
            
            if len(cur_scores) != 0:
                params[ch,:] = lows.mle_new(cur_scores,0)
        return params
    
    
    def __get_em_params(self, charges, labels, scores, fixed_pars=[], outname="em"):
        """get parameters from EM-based PSMs"""
        stats = np.zeros((10,5))
        null_params = np.zeros((10,2))
        
        chars = [2,3,4]
        fig, ax = plt.subplots(1, 3, figsize=(6, 2))
        idx_labels = set(np.where(labels != 4)[0])
        
        for idx in range(3):
            ch = chars[idx]
            
            idx_charges = set(np.where(charges == ch)[0])
            idx_shared = list(set.intersection(idx_charges, idx_labels))
            cur_scores = scores[idx_shared]
            
            #cur_tevs = df[(df.charge == ch) & (df.label != 4)]['tev'].to_numpy()
            if fixed_pars == []:
                params_em = ems.em_algorithm(cur_scores)
            else:
                params_em = ems.em_algorithm(cur_scores, fixed_pars[ch])
                
            ems.plot_em(ax[idx], cur_scores, params_em)
            stats[ch,:] = params_em
            null_params[ch,:] = params_em[:2]
            
        fig.tight_layout()
       
        fig.savefig(f"./graphs/{self.out}_EM_{outname}.png", dpi=600, bbox_inches='tight')
    
        return null_params, stats


    def __find_pi(self, ax_plot, data, params, plot=True):
        """find pi0 estimates for plotting the final models"""
        mu, beta = params
        axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
        kde = kde/auc(axes, kde)
        trunk = len(axes[axes < self.bic_cutoff])
        theory = lows.pdf_mubeta(axes, mu, beta, 0)
        err = 1000
        best_pi = 0

        for pi_0 in np.linspace(0, 1, 500):
            new_err = abs(auc(axes[:trunk], kde[:trunk]) - auc(axes[:trunk], pi_0*theory[:trunk]))
            if new_err < err:
                best_pi = pi_0
                err = new_err

        if plot:
            ax_plot.fill_between(axes, kde, alpha=0.2, color='#2CB199')
            ax_plot.plot(axes, kde, color='#2CB199')
            ax_plot.plot(axes, best_pi*theory, color='#D65215', linestyle='-')
            ax_plot.set_xlim(0.0, 0.6)
            ax_plot.set_ylim(0,20)
            ax_plot.set_xlabel("TEV")
            ax_plot.set_ylabel("density")

        return best_pi

    def __plot_top_models(self, tevs, mle_params, mm_params):
        
        def shift(arr, idx):
            return np.sign(arr[idx] - arr[idx+1])
        
        fig, ax = plt.subplots(1,3,figsize=(6, 2), constrained_layout=True)
        params = np.zeros((10,2))
        
        
        for order in range(3):
            top_hit = tevs[order][:,0]
            if mle_params[order][2].rvalue > 0.99 and np.mean(list(map(lambda x: shift(mle_params[order][0], x), range(9)))) < 0:
                #print(f"{order}, 'MLE'")
                best_mu, best_beta = self.__bic_optimization(top_hit, linreg=mle_params[order][2])

                if (best_mu < 0) or (best_beta < 0):
                    best_mu, best_beta = self.__bic_optimization(top_hit, mean_beta=np.mean(mm_params[order][1][-3:]))

            else:
                """
                mm_lr = st.linregress(mm_params[order][0][3:], mm_params[order][1][3:])
                print(mm_lr)
                if abs(mm_lr.rvalue) >= 0.99:
                    print("MM LR")
                    best_mu, best_beta = qq_lr(top_hit, mm_lr)
                if abs(mm_lr.rvalue) < 0.99:"""
                print(f"{order}, 'MM'")
                best_mu, best_beta = self.__bic_optimization(top_hit, mean_beta=np.mean(mm_params[order][1][-3:]))
                    
            best_pi = self.__find_pi(ax[order], top_hit, (best_mu, best_beta))
            params[order+2,:] = [best_mu, best_beta]
            #print(best_mu, best_beta, best_pi)
            
        #fig.tight_layout()
     
        fig.savefig(f"./graphs/{self.out}_top_models.png", dpi=600, bbox_inches="tight")
        return params


    def __bic_diff(self, data, k, order, mle_p, mm_p):
        """calculate difference between BIC for MLE and MM models"""
        bic_mm = self.__get_bic(data, k, order, mm_p)
        bic_mle = self.__get_bic(data, k, order, mle_p)
        return 100*(bic_mm-bic_mle)/abs(bic_mle)


    def __plot_lower_models(self, tevs, mle_par, mm_par, idx):
        """plot models of lower order distributions"""

        no_hits = len(tevs[0][0,:])
        n_row = int(np.ceil(np.sqrt(no_hits)))
        n_col = int(np.ceil(no_hits/n_row))

        fig, axs = plt.subplots(n_row, n_col, figsize=(n_row*2, n_col*2), constrained_layout=True)

        charge = idx-2
        bic_diffs = []

        for hit in range(no_hits-1):

            cur_bic = self.__calculate_and_plot(tevs, charge, hit, n_col, axs, mle_par, mm_par)
            bic_diffs.append(cur_bic)

        self.__add_axis_labels(axs, n_col, n_row)

        fig.savefig(f"./graphs/{self.out}_lower_models_{idx-2}.png", dpi=600, bbox_inches="tight")

        self.__plot_bics(bic_diffs, idx-2)


    def __calculate_and_plot(self, tevs, charge, hit, n_cols, axs, mle_par, mm_par):
        """break down the code"""
        data = tevs[charge][:,hit+1]
        kde_sup, kde_org = self.__get_kde_density(data)
        row, col = divmod(hit, n_cols)

        axs[row, col].plot(kde_sup, kde_org, color='#2D58B8')

        mle_pars = self.__extract_pars(mle_par[charge], hit)
        mm_pars = self.__extract_pars(mm_par[charge], hit)

        self.__kde_plots(axs[row, col], kde_sup, mle_pars, hit+1, color='#D65215')
        self.__kde_plots(axs[row, col], kde_sup, mm_pars, hit+1, color='#2CB199')
        axs[row, col].set_ylim(0,)

        cur_bic_diff = self.__bic_diff(data, k=2, order=hit+1, mle_p=mle_pars, mm_p=mm_pars)

        return cur_bic_diff

    @staticmethod
    def __add_axis_labels(axs, n_col, n_row):

        for idx in range(n_col*n_row):
            if idx % n_col == 0:
                axs[divmod(idx, n_col)].set_ylabel("density")

            if divmod(idx, n_col)[0] == n_row-1:
                axs[divmod(idx, n_col)].set_xlabel("TEV")



    @staticmethod
    def __get_kde_density(data):
        return FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)


    @staticmethod
    def __kde_plots(ax_plot, kde_sup, pars, hit, color):
        mu_, beta = pars
        pdf_vals = lows.pdf_mubeta(kde_sup, mu_, beta, hit)
        ax_plot.plot(kde_sup, pdf_vals, color=color)


    @staticmethod
    def __extract_pars(pars, hit):
        return pars[0][hit], pars[1][hit]


    def __plot_bics(self, bic_diffs, idx):
        """plot BICs"""
        support = np.arange(len(bic_diffs)) + 2
        fig, axs = plt.subplots(figsize=(10, 5), constrained_layout=True)
        axs.scatter(support, bic_diffs, color='#2D58B8')
        axs.plot(support, bic_diffs, color='#2D58B8')
        axs.set_xlabel("order")
        axs.set_ylabel("relative BIC difference [%]")
        fig.savefig(f"./graphs/{self.out}_lower_models_BIC_{idx}.png", dpi=600, bbox_inches="tight")


    @staticmethod    
    def __scatter_params(params, outname="example"):

        x=3
        fig, ax = plt.subplots(figsize=(4,4))

        for par_pair in params:
            ax.scatter(par_pair[0][x:], par_pair[1][x:])

        ax.set_xlabel("mu")
        ax.set_ylabel("beta")
        ax.set_title("testing")
        ax.legend(['2+', '3+', '4+'])
        #fig.savefig(f'{outname}_params_scatter.png', dpi=400, bbox_inches='tight')


    def __plot_lower_hist(self, tev, params, alpha):
        fig, ax = plt.subplots(3,3, figsize=(4,4))
        sss =1
        for row in range(3):
            for col in range(3):
                self.__plot_fit(ax[row%3, col], tev[alpha][:,sss], params[alpha][0][sss], params[alpha][1][sss], sss, col='#2D58B8', frac=1, bins=500)
                sss += 1
        #fig.savefig('yeast_3Da_1Da_f_lowerhits.png', dpi=400, bbox_inches='tight')
        
        
      
    def __plot_orders(self, mle_params, mm_params):
        no_orders = 10
        fig, ax = plt.subplots(2,3, figsize=(6,3), constrained_layout=True)
        cs = ['#2D58B8', '#D65215', '#2CB199']
        print(mle_params)
        print(mm_params)
    
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

    
    def __plot_mubeta(self, mle_params, mm_params):
        no_orders = 10
        fig, ax = plt.subplots(1,3, figsize=(9,3))
    
        for row in range(3):
            mle_c = '#2D58B8'
            mm_c = '#D65215'
            mle_x, mle_y = mle_params[row][0][3:], mle_params[row][1][3:]
            mm_x, mm_y = mm_params[row][0][3:], mm_params[row][1][3:]

            ax[row].scatter(mle_x, mle_y, color=mle_c, marker='o', edgecolors='k',linewidths=0.5)
            ax[row].scatter(mm_x, mm_y, color=mm_c, marker='o', edgecolors='k',linewidths=0.5)

            #print(f"charge {row+2}, MLE params")
            self.__annotation(ax[row], mle_x, mle_y, mle_c)
            #print(f"charge {row+2}, MM params")
            self.__annotation(ax[row], mm_x, mm_y, mm_c)

            ax[row].set_xlabel(r"$\mu$")
            #if row == 0:
            ax[row].set_ylabel(r"$\beta$")
              
        
        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mubeta_params_numbered.png", dpi=600, bbox_inches="tight")

        fig, ax = plt.subplots(1,3, figsize=(9,3))
    
        for row in range(3):
            mle_c = '#2D58B8'
            mm_c = '#D65215'
            mle_x, mle_y = mle_params[row][0][3:], mle_params[row][1][3:]
            mm_x, mm_y = mm_params[row][0][3:], mm_params[row][1][3:]

            #mle_lr = st.linregress(mle_x, mle_y)
            #mm_lr = st.linregress(mm_x, mm_y)

            #ax[row].plot(mle_x[0]*mle_lr.slope + mle_lr.intercept, )

            ax[row].scatter(mle_x, mle_y, color=mle_c, marker='o', edgecolors='k',linewidths=0.5)
            ax[row].scatter(mm_x, mm_y, color=mm_c, marker='o', edgecolors='k',linewidths=0.5)

            #self.annotation(ax[row], mle_x, mle_y, mle_c)
            #self.annotation(ax[row], mm_x, mm_y, mm_c)

            ax[row].set_xlabel(r"$\mu$")
            #if row == 0:
            ax[row].set_ylabel(r"$\beta$")

        fig.tight_layout()
       
        fig.savefig(f"./graphs/{self.out}_mubeta_params_clean.png", dpi=600, bbox_inches="tight")
        return (mle_params, mm_params)

    
    def __annotation(self, ax, x, y, col):

        offset = 3
        for idx, pair in enumerate(zip(x,y)):
            ax.annotate(idx+offset, (pair[0], pair[1]-0.0002), color=col)

        #plot linreg
        #linreg = st.linregress(x, y)
        #print(linreg.rvalue)
        #xs = np.array([min(x), max(x)])
        #ymin, ymax = linreg.slope*xs + linreg.intercept
        #ax.plot(xs, [ymin, ymax], color=col)
        #ax.set_title(f"{linreg.rvalue}_{linreg.pvalue}")

        
        
    def __plot_mubeta_lr(self, mle_params, mm_params):
        
        fig, ax = plt.subplots(1,3, figsize=(6, 2))
        offset = 2
        
        for col in range(3):
            
            ax[col].scatter(mle_params[col][0][offset:], mle_params[col][1][offset:], marker='.', color='#2D58B8')
            ax[col].scatter(mm_params[col][0][offset:], mm_params[col][1][offset:], marker='.', color='#D65215')
            
            if col == 0:
                ax[col].set_ylabel(r"$\beta$")
                ax[col].set_xlabel(r"$\mu$")
            else:
                ax[col].set_xlabel(r"$\mu$")
        
        
        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mubeta_LR.png", dpi=600, bbox_inches="tight")
    
        
    

    def __fast_parse(self, paths, option='Tide'):
        """fast parsing of pepXML results"""

        data = deque()

        for path in paths:
            cur_file = pepxml.read(path)
            psms = filter(lambda x: 'search_hit' in x.keys(), cur_file)
            has_top_n = filter(lambda x: len(x['search_hit']) == self.top_n, psms)

            if option == 'Tide':
                scores = list(map(self.__get_tide_data, has_top_n))
            elif option == 'Comet':
                scores = list(map(self.__get_comet_data, has_top_n))

            data += scores

        tevs, charges = list(zip(*data))
        return np.array(tevs), np.array(charges)


    def __get_tide_data(self, row):
        """extract TEV scores from Tide search results"""
        charge = row['assumed_charge']
        scores = map(self.__get_tide_tev,
                    row['search_hit'])
        return list(scores), charge

    @staticmethod
    def __get_tide_tev(x):
        """convert Tide's p-value to TEV"""
        if x['search_score']['exact_pvalue'] >= 10e-16:
            return -0.02*np.log(x['search_score']['exact_pvalue']*x['num_matched_peptides']/1000)

        return -0.02*np.log(10e-16*x['num_matched_peptides']/1000)

    @staticmethod
    def __get_comet_data(row):
        """extract TEV scores from Comet search results"""
        scores = map(lambda x:
                    -0.02*np.log(x['search_score']['expect']/1000),
                    row['search_hit'])
        charge = row['assumed_charge']
        return list(scores), charge



    #1 objective 1: estimate parameters for each hit separately, then plot the linear regression
    @staticmethod
    def __find_mle_pars(scores):
        """estimate parameters using MLE"""

        length = len(scores[0,:])
        params = [None,]*(length-1)

        for hit in range(1, length): # skip first hit since it's a mixture
            cur_tev = scores[:, hit].astype('float64')
            cur_tev = cur_tev[cur_tev > 0]
            #cur_tev = sorted(cur_tev)
            le = len(cur_tev)
            cur_tev = cur_tev[int(le*0.01):int(le*0.99)]
            params[hit-1] = lows.mle_new(cur_tev, hit)
            #mus.append(cur_mu)
            #betas.append(cur_beta)

        return list(zip(*params))
    
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
    def __plot_fit(ax, arr, N0, a, alpha, col='#2D58B8', frac=1, bins=500):
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


    def __plot_params(self, n0, a, xxx=0):

        trim_n0 = list(n0)
        trim_a = list(a)
        linreg = st.linregress(trim_n0, trim_a)
        #print(linreg)

        fig = plt.figure(figsize=(4,4))
        plt.scatter(trim_n0, trim_a, marker='o', color='#2D58B8')
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
    def __get_mle_params(self, tevs, cutoff=3):

        mu, beta = self.__find_mle_pars(tevs)
        trim_n0 = mu[cutoff:]
        trim_a = beta[cutoff:]
        linreg = st.linregress(trim_n0, trim_a)

        return mu, beta, linreg

    #obtain method of moments parameters
    @staticmethod
    def __get_mm_params(tev):
        m1 = []
        m2 = []
        all_orders = len(tev[0,:])
        for order in range(1,all_orders):
            cur_m1, cur_m2 = lows.mm_estimator(tev[:,order], order)
            m1.append(cur_m1)
            m2.append(cur_m2)
        return m1, m2


###########################################################


################# BIC OPTIMIZATION ###################


    def __bic_optimization(self, tev, **kwargs):
        """BIC optimization for linear regression mode parameters"""
        errors = []
        mu_range = np.linspace(0.05, 0.4, 500)

        if 'linreg' in kwargs:
            linreg = kwargs['linreg']
            for cur_mu in mu_range:
                cur_beta = cur_mu*linreg.slope + linreg.intercept
                diffs = self.__get_bic(tev, 1, 0, [cur_mu, cur_beta])
                errors.append(diffs)

        elif 'mean_beta' in kwargs:
            opt_beta = kwargs['mean_beta']
            for cur_mu in mu_range:
                diffs = self.__get_bic(tev, 1, 0, [cur_mu, opt_beta])
                errors.append(diffs)

        opt_idx = errors.index(min(errors))
        opt_mu = mu_range[opt_idx]

        if 'linreg' in kwargs:
            opt_beta = opt_mu*linreg.slope + linreg.intercept

        return opt_mu, opt_beta

    ###########################################################


   ############### VALIDATION with BH procedure #############
    @staticmethod
    def __get_val_data(row, pars):

        tev = -0.02 * np.log((row['search_hit'][0]['search_score']['expect']) / 1000)
        ch = int(row['assumed_charge'])

        if ch not in [2,3,4]:
            pv = 1
        else:
            pv = 1 - lows.mubeta_cdf(tev, pars[ch][0], pars[ch][1])

        return tev, ch, pv

   #process randoms
    def __parse_data(self, idx, keywords, paths, labels, pars):
        
        keyword = keywords[idx]
        label_value = labels[idx]
        rand_paths = list(filter(lambda x: keyword in x, paths))
        items = deque()
        
        for pepxml_file in rand_paths:
            cur_file = pepxml.read(pepxml_file)
            data = list(cur_file.map(self.__get_val_data, args=(pars,)))
            items += data

        scores = [x[0] for x in items]
        charges = [x[1] for x in items]
        pvs = [x[2] for x in items]
        labels = list(label_value*np.ones(len(items)))
        
        return scores, charges, pvs, labels
    
    def __faster_validation(self, paths, pars):
       
        pvs = deque()
        ground_truth = deque()
        charges = deque()
        scores = deque()        
            
        keywords = ["random", "decoy", "pos"]
        labels = [0, 4, 1]

        big_data = list(map(fu.partial(self.__parse_data, keywords=keywords,
                                            paths=paths, labels=labels, 
                                            pars=pars), np.arange(3)))

        for item in big_data:
            scores += item[0]
            charges += item[1]
            pvs += item[2]
            ground_truth += item[3]
            
    
        #df = pd.DataFrame(np.array([pvs, ground_truth, charges, scores]).T) 
        #df.head()
        #df.columns = ['pv_low', 'label', 'charge', 'tev']
        return np.array(scores), np.array(charges), np.array(pvs), np.array(ground_truth)
    
   
    @staticmethod
    def __validation_df_random(paths, pars):
        
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
    def __validation_df(pepxml_file, ref_dict, params):
        
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
    def __get_bh(df, bh, pv_name='pv_low'):

        df.loc[:,'bh'] = bh.values
        finaldf = df[df[pv_name] <= df['bh']]

        return finaldf

    #when FDR is calculated using BH method
    def __reps_single(self, reps, length, pvs, labels, len_correct):
        
        random.seed()
        new_sel = random.choices(length, k=len(length))
        
        new_pvs = pvs[new_sel]
        new_labels = labels[new_sel]
        
        fdr, fdp, tp = self.fdr_lower(new_pvs, new_labels, len_correct)
        
        return fdr, fdp, tp


    def __bootstrap_fdr(self, reps, labels, pvs, len_correct):
        """generate bootstrapped FDP estimates, get subsample from dataframe
            and calculate the stats for it, repeat"""

        length = np.arange(len(labels))
        #fdrs = np.linspace(0.0001, 0.1, 100)
        fdrs = np.zeros((self.reps, 100))
        fdps = np.zeros((self.reps, 100))
        tps = np.zeros((self.reps, 100))

        data = list(map(fu.partial(self.__reps_single, length=length,
                                            pvs=pvs, labels=labels, 
                                            len_correct=len_correct), np.zeros(reps)))
        fdrs = [x[0] for x in data]
        fdps = [x[1] for x in data]
        tps = [x[2] for x in data]

        return fdrs, fdps, tps
    
    @staticmethod
    def map_add_pvs(idx, scores, charges, pars):
        
        tev = scores[idx]
        ch  = charges[idx]
        pv = 1 - lows.mubeta_cdf(tev, pars[ch][0], pars[ch][1])
        return pv
    
    def faster_add_pvs(self,scores, charges, params):
        
        indices = np.arange(len(scores))
        pvs = list(map(fu.partial(self.map_add_pvs, 
                                  scores=scores, 
                                   charges=charges, 
                                   pars=params), indices))
        return np.array(pvs)
        
        
    """
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
    """
    
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
        
    
    
    @staticmethod
    def get_fdr(fdr, pvs, labels, len_correct, idx_for_bh):
        
        bh = idx_for_bh*fdr/len(pvs)
        
        adj_index = np.where(pvs <= bh)[0]
        len_accepted = len(adj_index)
        adj_labels = labels[adj_index]
               
        if len_accepted == 0: len_accepted = 1
        if len_correct == 0: len_correct = 1
        
        len_tps = len(adj_labels[adj_labels == 1])
        
        fdp = 1-len_tps/len_accepted
        #dec = 2*len(ch3[ch3['label'] == 4])/len(ch3)
        tp = len_tps/len_correct
        
        return fdp, tp
        
    
    
   #generate the data of FDP and TP for the selected FDR range
    def fdr_lower(self, pvs, labels, len_correct):
        
        fdrs = np.linspace(0.0001, 0.1, 100)
        
        #select only target PSMs of the desired charge
        sorted_index = np.argsort(pvs)        
        idx_for_bh = np.arange(len(pvs)) + 1
        sorted_pvs = pvs[sorted_index]
        sorted_labels = labels[sorted_index]
     
        #faster code for fdr calculation
        data = list(map(fu.partial(self.get_fdr, pvs=sorted_pvs,
                                   labels=sorted_labels,
                                   len_correct=len_correct,
                                   idx_for_bh=idx_for_bh), fdrs))
        
        fdps = [x[0] for x in data]
        tps = [x[1] for x in data]

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

       ############### VALIDATION with PeptideProphet ###########
    @staticmethod
    def peptideprophet_validation(interact_file, no_files, ref_peps):

        length = 500000
        pvs = -1*np.ones(length)
        labels = np.zeros(length)
        charges = np.zeros(length)
        tevss = np.zeros(length)
        k=0
        new_seqs = deque()



        d = pepxml.read(interact_file)

        for el in d:
        # if 'DECOY' in el['search_hit'][0]['proteins'][0]['protein']:
        #     continue
            if 'search_hit' in el.keys():
                p_v = el['search_hit'][0]['analysis_result'][0]['peptideprophet_result']['probability']
                spec = el['spectrum']
                fval = el['search_hit'][0]['search_score']['expect']
                fval = -0.02 * np.log(fval / 1000.)
                pep = el['search_hit'][0]['peptide']
                new_seq = pep.replace('I', 'X').replace('L', 'X')

        
                if new_seq in ref_peps:
                    label = 1
                else:
                    label = 0

                
                    
                pvs[k] = p_v
                tevss[k] = fval
                labels[k] = label
                new_seqs.append(new_seq)
                
                k +=1
            
        

        df = pd.DataFrame(np.array([pvs, tevss, labels]).T)
        df.columns = ['PP_pval', 'TEV', 'label']
        df = df[df['PP_pval'] != -1]
        df['peptide'] = new_seqs
        #df['spectrum'] = specs

        df = df.sort_values('PP_pval', inplace=False, ascending=True)
        df = df.reset_index(drop=True)
        df.index += 1

        tree = ET.parse(interact_file)
        root = tree.getroot()

        #index: FDR threshold
        #34 0.001
        #45 0.01
        #47 0.02
        #50 0.04
        #51 0.05
        #49 0.03

        fdr_indices = [34, 40, 45, 46, 47, 48, 49, 50, 51]
        thresholds = list(map(lambda x: float(root[0][0][int(no_files)][x].attrib['min_prob']), fdr_indices))

        return df, thresholds
    
    def plot_peptideprophet_validation_results(self, synth_pep_list):

        def replace_IL(pep):
            return pep.replace('I', 'X').replace('L', 'X')

        def get_stats(df, ths):
            fdps = []
            tps = []
            for i in ths:
                filtered = df[df['PP_pval'] >= i]
                fdps.append(len(filtered[filtered.label==0])/len(filtered))
                tps.append(len(set(filtered[filtered.label==1]["peptide"].values)))
                
            return fdps, tps

        peps = pd.read_csv(synth_pep_list, header=None)
        fdrs = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
        colors = ["royalblue", "orange", "green"]
        styles = ["-", "--", "-."]
        fig, ax = plt.subplots(1,2, figsize=(7, 3.5))
        ax[0].plot([0, 0.05], [0, 0.05], color="grey")

        for idx, name_id in enumerate(["26", "36", "42"]):
            print(f"this is {name_id}...")

            ref_peps = set(peps[peps[0].str.contains(f"first_pool_{name_id}")][1].values)
            x_peps = list(map(lambda x: replace_IL(x), ref_peps))

            print("this is td param")
            df, ths = self.peptideprophet_validation(f"/data/dominik/pp_cdd_validation/interact-{name_id}_td_raw_par.pep.xml", 1, x_peps)
            fdps, tps = get_stats(df, ths)
            #ax[0].scatter(fdrs,fdps, color=colors[1])
            ax[0].plot(fdrs, fdps, color=colors[1], linestyle=styles[idx])
            #ax[1].scatter(fdrs,tps, color=colors[1])
            ax[1].plot(fdrs, tps, color=colors[1], linestyle=styles[idx])

            print("this is td nonparam")
            df, ths = self.peptideprophet_validation(f"/data/dominik/pp_cdd_validation/interact-{name_id}_td.pep.xml", 1, x_peps)
            fdps, tps = get_stats(df, ths)
            #ax[0].scatter(fdrs,fdps, color=colors[2])
            ax[0].plot(fdrs, fdps, color=colors[2], linestyle=styles[idx])
            #ax[1].scatter(fdrs,tps, color=colors[2])
            ax[1].plot(fdrs, tps, color=colors[2], linestyle=styles[idx])

            print("this is cdd")
            df, ths = self.peptideprophet_validation(f"/data/dominik/pp_cdd_validation/interact-cdd_{name_id}.pep.xml", 1, x_peps)
            fdps, tps = get_stats(df, ths)
            #ax[0].scatter(fdrs,fdps, color=colors[0])
            ax[0].plot(fdrs, fdps, color=colors[0], linestyle=styles[idx])
            #ax[1].scatter(fdrs,tps, color=colors[0])
            ax[1].plot(fdrs, tps, color=colors[0], linestyle=styles[idx])


        ax[0].set_xlabel("FDR threshold")
        ax[0].set_ylabel("FDP")
        ax[1].set_xlabel("FDR threshold")
        ax[1].set_ylabel("number of peptides identified")

        fig.tight_layout()

        plt.savefig("peptideprophet_validation.png", dpi=600)




                            
