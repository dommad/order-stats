"""Full analysis of pepxml file using lower order statistics"""
import random
from collections import deque
import functools as fu
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from xml.etree import ElementTree as ET
from pyteomics import pepxml, mzid
from KDEpy import FFTKDE
from sklearn.metrics import auc
import order_formulae as of

warnings.filterwarnings("ignore")
ofs = of.Tools()

TH_N0 = 1000.
TH_MU = 0.02*np.log(TH_N0)
TH_BETA = 0.02

class Analyze:
    """Execute FDR estimation using lower-order PSMs"""

    def __init__(self, outname, bic_cutoff=0.17, lr_cutoff=(5,10)):
        self.params_est = []
        self.out = outname
        self.len_correct = 0
        self.bic_cutoff = bic_cutoff
        self.tevs = []
        self.lr_cutoff = lr_cutoff
        self.all_charges = [2,3,4]


    def run_estimation(self, input_paths, pars_outname, mode='Tide', top_n=30):
        """Estimate parameters of top null model using lower order TEV distributions"""
        if mode in {'Tide', 'Comet'}:
            self.tevs, charges = self.__parse_tide_comet(input_paths, top_n, mode)
        elif mode == 'MSGF-tsv':
            self.tevs, charges = self.__parse_msgf_tsv(input_paths[0])
        elif mode == 'MSGF-mzid':
            self.tevs, charges = self.__parse_msgf_mzid(input_paths[0])

        data = list(map(fu.partial(self.__get_mle_mm_pars, charges), self.all_charges))
        self.tevs, mle_pars, mm_pars = list(zip(*data))
        mm_pars = np.nan_to_num(np.array(mm_pars))

        _ = self.__plot_mubeta(mle_pars, mm_pars)
        self.__plot_mubeta_single(mle_pars, 'mle')
        self.__plot_mubeta_single(mm_pars, 'mm')
        self.params_est = self.__alt_top_models(mle_pars, mm_pars, len(self.all_charges))

        for charge in self.all_charges:
            idx = charge-2
            self.__plot_lower_models(mle_pars, mm_pars, idx)

        self.__export_pars_to_txt(pars_outname)
        return self.params_est, mle_pars, mm_pars, self.tevs, charges


    def __export_pars_to_txt(self, pars_outname):
        """export params to txt for modified PeptideProphet (mean & std)"""
        params = pd.DataFrame(self.params_est[1:])
        params[0] = params[0] + params[1]*np.euler_gamma
        params[1] = np.pi/np.sqrt(6)*params[1]
        params.to_csv(f"{pars_outname}.txt", sep=" ", header=None, index=None)


    def __get_mle_mm_pars(self, charges, sel_charge):
        """calculate MLE and MM parameters using the data"""
        sel_scores = self.tevs[np.where(charges == sel_charge)[0]]
        if len(sel_scores) == 0:
            sel_scores = self.tevs[np.where(charges == 2)[0]]
        mle_pars = tuple(self.__get_mle_pars(sel_scores))
        mm_pars = tuple(self.__get_mm_pars(sel_scores))
        return sel_scores, mle_pars, mm_pars


    def __get_lr_mean_beta(self, pars, first_idx=4, last_beta_idx=-3):
        """get linear regression parameters and mean beta for further processing"""
        mu_idx, beta_idx = [0,1]
        first_idx = self.lr_cutoff[0]
        linreg = st.linregress(pars[mu_idx][first_idx:],
                                pars[beta_idx][first_idx:]) # drop first 3 points
        mean_beta = np.mean(pars[beta_idx][last_beta_idx:]) # use last 3 data points
        return linreg, mean_beta


    def __get_bic(self, data, k, order, params):
        """calculate BIC for lower section of the distribution"""
        data = data[data < self.bic_cutoff]
        log_like = ofs.log_like_mubeta(np.log(params), data, order)
        bic = k*np.log(len(data)) - 2*log_like
        return bic


    def __get_alt_params(self, data, estim_mode):
        """generate LR and mean beta params for given data"""
        linreg, beta = estim_mode
        lr_mu, lr_beta = self.bic_optimize(data, linreg=linreg)
        mean_mu, mean_beta = self.bic_optimize(data, mean_beta=beta)
        return lr_mu, lr_beta, mean_mu, mean_beta


    def __alt_top_models(self, mle_pars, mm_pars, charge_no):
        """estimate parameters of top models using lower order distributions"""
        fig, axes = plt.subplots(1, charge_no, figsize=(2*charge_no, 2))
        fin_pars = np.zeros((10,2))

        # mode_dict  = {0: "MLE LR", 1: "MLE mean", 2: "MM LR", 3: "MM mean"}

        for ch_idx in range(charge_no):

            top_hit = self.tevs[ch_idx][:,0]
            top_hit = top_hit[top_hit > 0.01] # dommad: don't count scores at 0 and below
            mode_pars = []
            bics = []
            mode_pars.append(self.__get_lr_mean_beta(mle_pars[ch_idx]))
            mode_pars.append(self.__get_lr_mean_beta(mm_pars[ch_idx]))
            tmp_pars = []

            for mode in mode_pars:
                tmp_pars, bics = self.__get_bics_data(top_hit, mode, tmp_pars, bics)

            best_idx = bics.index(min(bics))
            # print(bics)
            # print(f"best idx for charge {ch_idx+2} is {mode_dict[best_idx]}")
            _ = self.__find_pi(axes[ch_idx], top_hit, tmp_pars[best_idx]) # for plotting
            fin_pars[ch_idx+2,:] = tmp_pars[best_idx]

        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_alt_top_models.png", dpi=600, bbox_inches="tight")
        return fin_pars


    def __get_bics_data(self, top_hit, mode, tmp_pars, bics):
        """obtain BICs and temporary parameters"""
        lr_mu, lr_beta, m_mu, m_beta = self.__get_alt_params(top_hit, mode)
        print(lr_mu, lr_beta, m_mu, m_beta)

        bic_lr = self.__get_bic(top_hit, 2, 0, [lr_mu, lr_beta]) #dommad removed 0.8
        bic_m = self.__get_bic(top_hit, 2, 0, [m_mu, m_beta])

        tmp_pars.append([lr_mu, lr_beta])
        tmp_pars.append([m_mu, m_beta])

        if (lr_mu < 0) or (lr_beta < 0):
            bic_lr = 1e6

        bics.append(bic_lr)
        bics.append(bic_m)

        return tmp_pars, bics


    def __get_dec_pars(self, data):
        pars = np.zeros((7,2))
        scores, chars = data

        for charge in self.all_charges:
            cur_mask = np.where(chars == charge)[0]
            cur_scores = scores[:,0][cur_mask]
            if len(cur_scores) < 100:
                pars[charge,:] = (0,0)
                continue
            pars[charge,:] = st.gumbel_r.fit(cur_scores)
        return pars


    def execute_validation(self, pepxml_f, top_n=10, reps=500, ext_params="", dec_paths=(), mode='Tide'):
        """read the pepxml, automatically add the p-values based on lower order estimates"""

        if ext_params != "":
            self.params_est = ext_params

        scrs, chars, low_pvs, coute_pvs, lbls = self.__parse_get_pvals(pepxml_f, self.params_est, option=mode)

        dec_data = self.__parse_tide_comet(dec_paths, top_n, option=mode, decoy=bool(mode == 'Comet'))

        dec_pars = self.__get_dec_pars(dec_data)
        cdd_pars = pd.read_csv('./cdd_params.txt', header=None).to_numpy()

        # this pi_0 estimate is calculated according to Jiang and Doerge (2008)
        # pi_0 = PiZeroEstimator().find_optimal_pi0(low_pvs, 10)

        pi_0 = len(low_pvs[np.where(lbls == 0)])/len(low_pvs[np.where(lbls != 4)])
                                            

        dec_pvs = self.__add_pvs(scrs, chars, dec_pars)
        cdd_pvs = self.__add_pvs(scrs, chars, cdd_pars)

        all_boot_stats = deque()
        idx_non_dec = list(set(np.where(lbls != 4)[0]))

        cur_labels = lbls[idx_non_dec]
        cur_lower = low_pvs[idx_non_dec]
        cur_decoy = dec_pvs[idx_non_dec]
        cur_cdd = cdd_pvs[idx_non_dec]
        cur_coute = coute_pvs[idx_non_dec]
        self.len_correct = len(cur_labels[cur_labels == 1])

        print("here")
        stats_low = self.__bootstrap(reps, cur_labels, cur_lower)
        print("here")
        stats_dec = self.__bootstrap(reps, cur_labels, cur_decoy)
        print("here")
        stats_cdd = self.__bootstrap(reps, cur_labels, cur_cdd)
        print("here")
        stats_coute = self.__bootstrap(reps, cur_labels, cur_coute)

        all_boot_stats.append([stats_low, stats_dec, stats_cdd, stats_coute])

        fig, axs = plt.subplots(1, 2, figsize=(6,3))
        self.__plot_boot_fdrs(axs[0], all_boot_stats, pi_0)
        self.__plot_boot_tps(axs[1], all_boot_stats)
        fig.tight_layout()

        fig.savefig(f"./graphs/{self.out}_validation.png", dpi=600, bbox_inches='tight')

        return all_boot_stats


    def __bootstrap(self, reps, labels, pvs):
        """calculate the consolidated stats"""
        bootstrap_data = self.__bootstrap_fdr(reps, labels, pvs, self.len_correct)
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

    def __plot_boot_fdrs(self, axs, all_stats, pi_0):
        """plotting bootstrap FDP vs FDR"""

        cs_ = ['#2D58B8', '#D65215', '#2CB199', '#7600bc']

        for method in range(len(all_stats[0])):
            fdrs = all_stats[0][method][0][0,:]
            fdps = all_stats[0][method][1]
            if method == 0:
                self.__plot_fdr_stat(axs, pi_0*fdrs, np.array(fdps), cs_[method], xy_=1)
            else:
                self.__plot_fdr_stat(axs, pi_0*fdrs, np.array(fdps), cs_[method])

        axs.set_xlabel("FDR")
        axs.set_ylabel("FDP")


    def __plot_boot_tps(self, axs, all_stats):
        """plot boostrap # identified PSMs"""
        cs_ = ['#2D58B8', '#D65215', '#2CB199', '#7600bc']

        for method in range(len(all_stats[0])):
            fdrs = all_stats[0][method][0][0,:]
            tps = all_stats[0][method][2]
            if method == 0:
                self.__plot_fdr_stat(axs, fdrs, tps, cs_[method], axis_t="TPR")
            else:
                self.__plot_fdr_stat(axs, fdrs, tps, cs_[method], axis_t="TPR")

        axs.set_xlabel("FDR")
        axs.set_ylabel("Correctly identified PSMs")
        axs.set_ylim(0,17000)
        axs.set_xlim(0,0.1)


    #plot the FDP vs FDR results of validation
    @staticmethod
    def __plot_fdr_stat(axs, fdrs, fdp_stats, col, xy_=False, axis_t='FDP'):
        """plot FDP vs FDR - results of validation"""
        #fdrs = np.linspace(0.0001, 0.1, 100)

        if xy_:
            axs.plot([0.0001,0.1], [0.0001, 0.1], c='gray')

        axs.plot(fdrs, fdp_stats[0,:], color=col, linewidth=2)
        axs.fill_between(fdrs, fdp_stats[0,:], fdp_stats[2,:], alpha=0.2, color=col)
        axs.fill_between(fdrs, fdp_stats[0,:], fdp_stats[1,:], alpha=0.2, color=col)


        if axis_t == 'TPR':
            axs.set_xlim(-0.001, 0.1)
            axs.set_ylim(-0.01,)
        else:
            axs.set_xlim(-0.001, 0.1+0.001)
            axs.set_ylim(-0.001, 0.1+0.001)


    def __find_pi(self, axs, data, pars, plot=True):
        """find pi0 estimates for plotting the final models"""
        mu_, beta = pars
        axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
        kde = kde/auc(axes, kde)
        trunk = len(axes[axes < self.bic_cutoff])
        theory = ofs.pdf_mubeta(axes, mu_, beta, 0)
        err = 1000
        best_pi = 0

        for pi_0 in np.linspace(0, 1, 500):
            new_err = abs(auc(axes[:trunk], kde[:trunk]) - auc(axes[:trunk], pi_0*theory[:trunk]))
            if new_err < err:
                best_pi = pi_0
                err = new_err

        if plot:
            axs.fill_between(axes, kde, alpha=0.2, color='#2CB199')
            axs.plot(axes, kde, color='#2CB199')
            axs.plot(axes, best_pi*theory, color='#D65215', linestyle='-')
            axs.set_xlim(0.0, 0.6)
            axs.set_ylim(0,20)
            axs.set_xlabel("TEV")
            axs.set_ylabel("density")

        return best_pi


    def compare_density_auc(self, data, params, hit, idx):
        """Compare theoretical and empirical densities"""
        mle_pars = self.__extract_pars(params[idx], hit)
        mu_, beta = mle_pars
        axes, kde = FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)
        kde_auc = auc(axes, kde)
        theory = ofs.pdf_mubeta(axes, mu_, beta, hit)
        theory_auc = auc(axes, theory)
        return abs(theory_auc - kde_auc)/kde_auc


    def __plot_lower_models(self, mle_par, mm_par, idx):
        """plot models of lower order distributions"""

        #idx = charge -2
        bic_diffs = []
        n_row = 3
        n_col = 3

        fig1, ax1 = plt.subplots(n_row, n_col, figsize=(n_row*2, n_col*2), constrained_layout=True)
        fig2, ax2 = plt.subplots(n_row, n_col, figsize=(n_row*2, n_col*2), constrained_layout=True)
        hit = 0
        for row in range(3):
            for col in range(3):
                cur_bic = self.__add_lower_plot(idx, hit, ax1[row, col], mle_par, mm_par)
                self.__add_pp_plot(idx, hit, ax2[row, col], [mle_par, mm_par])
                #cur_bic = self.compare_density_auc(self.tevs[idx][:,hit], mle_par, hit, idx)
                bic_diffs.append(cur_bic)
                hit += 1

        self.__add_axis_labels(ax1, n_col, n_row, mode='density')
        self.__add_axis_labels(ax2, n_col, n_row, mode='PP')
        fig1.savefig(f"./graphs/{self.out}_lower_models_{idx}.png", dpi=600, bbox_inches="tight")
        fig2.savefig(f"./graphs/{self.out}_pp_plots_{idx}.png", dpi=600, bbox_inches="tight")

        self.__plot_bics(bic_diffs, idx)

    def __add_pp_plot(self, idx, hit, axs, pars_list):
        """break down the code"""
        data = sorted(self.tevs[idx][:, hit+1])
        data = np.array(data)
        data = data[data>0.01] #dommad

        mle_par, mm_par = pars_list
        mle_pars = self.__extract_pars(mle_par[idx], hit)
        mm_pars = self.__extract_pars(mm_par[idx], hit)

        mm_qq = ofs.universal_cdf(data, mm_pars[0], mm_pars[1], hit+1)
        mle_qq = ofs.universal_cdf(data, mle_pars[0], mle_pars[1], hit+1)
        emp_qq = np.arange(1, len(data)+1)/len(data)
        axs.scatter(mle_qq, emp_qq, color='#D65215', s=1)
        axs.scatter(mm_qq, emp_qq, color='#2CB199', s=1)
        axs.plot([0,1], [0,1], color='k')


    def __add_lower_plot(self, idx, hit, axs, mle_par, mm_par):
        """break down the code"""

        kde_sup, kde_org = self.__get_hit_scores(idx, hit)

        if len(kde_org) == 0:
            return 0

        axs.plot(kde_sup, kde_org, color='#2D58B8')

        mle_pars = self.__extract_pars(mle_par[idx], hit)
        mm_pars = self.__extract_pars(mm_par[idx], hit)

        self.__kde_plots(axs, kde_sup, mle_pars, hit+1, color='#D65215')
        self.__kde_plots(axs, kde_sup, mm_pars, hit+1, color='#2CB199')
        axs.set_ylim(0,)

        cur_bic_diff = self.__bic_diff(idx, k=2, order=hit+1, params=[mle_pars, mm_pars])
        return cur_bic_diff

    def __bic_diff(self, idx, k, order, params):
        """calculate difference between BIC for MLE and MM models"""
        mle_p, mm_p = params
        data = self.tevs[idx][:, order]
        _ = self.__get_bic(data, k, order, mm_p)
        bic_mle = self.__get_bic(data, k, order, mle_p)
        # return 100*(bic_mm-bic_mle)/abs(bic_mle)
        return bic_mle


    def __get_hit_scores(self, idx, hit):
        """get hit scores from TEV array"""
        data = self.tevs[idx][:, hit+1]
        data = data[data > 0.01] # dommad use scores above 0
        if len(data) == 0:
            return (), ()
        kde_sup, kde_den = self.__get_kde_density(data)
        return kde_sup, kde_den


    @staticmethod
    def __add_axis_labels(axs, n_col, n_row, mode='density'):

        if mode == 'density':
            ylab = 'density'
            xlab = 'TEV'
        elif mode == 'PP':
            ylab = 'empirical CDF'
            xlab = 'theoretical CDF'

        for idx in range(n_col*n_row):
            if idx % n_col == 0:
                axs[divmod(idx, n_col)].set_ylabel(ylab)

            if divmod(idx, n_col)[0] == n_row-1:
                axs[divmod(idx, n_col)].set_xlabel(xlab)


    @staticmethod
    def __get_kde_density(data):
        return FFTKDE(bw=0.0005, kernel='gaussian').fit(data).evaluate(2**8)


    @staticmethod
    def __kde_plots(ax_plot, kde_sup, pars, hit, color):
        mu_, beta = pars
        pdf_vals = ofs.pdf_mubeta(kde_sup, mu_, beta, hit)
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


    def __plot_mubeta_single(self, pars, outn='mle'):

        fig, axs = plt.subplots(1,3, figsize=(9,3))
        l_lim, u_lim = self.lr_cutoff
        cs_ = ['#2D58B8', '#D65215']

        for row in range(3):
            mle_x, mle_y = pars[row][0][l_lim:u_lim], pars[row][1][l_lim:u_lim]

            axs[row].scatter(mle_x, mle_y, color=cs_[0], marker='o', edgecolors='k',linewidths=0.5)
            linreg = st.linregress(mle_x, mle_y)
            lr_x = np.array([min(TH_MU, min(mle_x)), max(mle_x)])
            axs[row].plot(lr_x, lr_x*linreg.slope + linreg.intercept)
            axs[row].scatter([TH_MU], [TH_BETA], color='orange')
            #axs[row].set_title(f"pvalue = {linreg.pvalue}")
            #self.__annotation(ax[row], mle_x, mle_y, cs_[0])
            axs[row].set_xlabel(r"$\mu$")
            axs[row].set_ylabel(r"$\beta$")

        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_{outn}_mubeta_params.png", dpi=600, bbox_inches="tight")


    def __plot_mubeta(self, mle_params, mm_params):
        """plot mu-beta plots"""
        fig, axs = plt.subplots(1,3, figsize=(9,3))

        for row in range(3):
            mle_c = '#2D58B8'
            mm_c = '#D65215'
            mle_x, mle_y = mle_params[row][0][3:], mle_params[row][1][3:]
            mm_x, mm_y = mm_params[row][0][3:], mm_params[row][1][3:]

            axs[row].scatter(mle_x, mle_y, color=mle_c, marker='o', edgecolors='k',linewidths=0.5)
            axs[row].scatter(mm_x, mm_y, color=mm_c, marker='o', edgecolors='k',linewidths=0.5)

            self.__annotation(axs[row], mle_x, mle_y, mle_c)
            self.__annotation(axs[row], mm_x, mm_y, mm_c)

            axs[row].set_xlabel(r"$\mu$")
            axs[row].set_ylabel(r"$\beta$")

        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mubeta_params_numbered.png", dpi=600)

        fig, axs = plt.subplots(1,3, figsize=(9,3))

        for row in range(3):
            mle_c = '#2D58B8'
            mm_c = '#D65215'
            mle_x, mle_y = mle_params[row][0][3:], mle_params[row][1][3:]
            mm_x, mm_y = mm_params[row][0][3:], mm_params[row][1][3:]

            # mle_lr = st.linregress(mle_x, mle_y)
            # mm_lr = st.linregress(mm_x, mm_y)

            # ax[row].plot(mle_x[0]*mle_lr.slope + mle_lr.intercept, )

            axs[row].scatter(mle_x, mle_y, color=mle_c, marker='o', edgecolors='k',linewidths=0.5)
            axs[row].scatter(mm_x, mm_y, color=mm_c, marker='o', edgecolors='k',linewidths=0.5)

            # self.annotation(ax[row], mle_x, mle_y, mle_c)
            # self.annotation(ax[row], mm_x, mm_y, mm_c)

            axs[row].set_xlabel(r"$\mu$")
            # if row == 0:
            axs[row].set_ylabel(r"$\beta$")

        fig.tight_layout()
        fig.savefig(f"./graphs/{self.out}_mubeta_params_clean.png", dpi=600)
        return (mle_params, mm_params)

    def __annotation(self, axs, xs_, ys_, col):

        offset = 3
        for idx, pair in enumerate(zip(xs_,ys_)):
            axs.annotate(idx+offset, (pair[0], pair[1]-0.0002), color=col)


    def __parse_tide_comet(self, paths, top_n, option='Tide', decoy=False):
        """fast parsing of pepXML results from Tide or Comet"""
        data = deque()

        for path in paths:
            cur_file = pepxml.read(path)
            psms = filter(lambda x: 'search_hit' in x.keys(), cur_file)
            has_top_n = filter(lambda x: len(x['search_hit']) == top_n, psms)

            if decoy:
                has_top_n = filter(lambda x: 'DECOY' in x['search_hit'][0]['proteins'][0]['protein'], has_top_n)

            if option == 'Tide':
                scores = list(map(self.__get_tide_scores, has_top_n))
            elif option == 'Comet':
                scores = list(map(self.__get_comet_scores, has_top_n))

            data += scores

        tevs, charges = list(zip(*data))
        tevs = np.nan_to_num(np.array(tevs))
        return tevs, np.array(charges)


    def __get_tide_scores(self, spectrum):
        """extract TEV scores from Tide search results"""
        charge = spectrum['assumed_charge']
        scores = map(self.__get_tide_tev,
                    spectrum['search_hit'])
        return list(scores), charge

    @staticmethod
    def __get_tide_tev(hit):
        """convert Tide's p-value to TEV"""
        num_match = hit['num_matched_peptides']
        p_val = hit['search_score']['exact_pvalue']
        return -TH_BETA*np.log(max(p_val, 10e-16)*num_match/TH_N0)

    @staticmethod
    def __get_comet_scores(row):
        """extract TEV scores from Comet search results"""
        scores = map(lambda x:
                    -TH_BETA*np.log(x['search_score']['expect']/TH_N0),
                    row['search_hit'])
        charge = row['assumed_charge']
        return list(scores), charge


    def __parse_msgf_tsv(self, input_path):
        """Parse MSGF+ search results"""
        data = pd.read_csv(input_path, sep='\t')
        scans = set(data.ScanNum)
        tevs = list(map(fu.partial(self.__get_msgf_tsv_tev, dat=data, no_hits=30), scans))
        charges = list(map(lambda x: data[data.ScanNum == x]['Charge'].values[0], scans))
        return np.array(tevs), np.array(charges)

    @staticmethod
    def __get_msgf_tsv_tev(scan, dat, no_hits):
        """parse MSGF+ scores fast"""
        cur_scores = [0,]*no_hits
        cur_hits = set(dat[dat.ScanNum == scan]['EValue'])
        cur_hits = np.array(sorted(list(cur_hits)))
        cur_hits = -TH_BETA*np.log(cur_hits/TH_N0)
        if len(cur_hits) > no_hits:
            return cur_hits[:no_hits]
        cur_scores[:len(cur_hits)] = cur_hits
        return cur_scores

    @staticmethod
    def __get_msgf_mzid_tev(spec):
        """Extract TEVs from MSGF+ output (mzid)"""
        charge = spec['SpectrumIdentificationItem'][0]['chargeState']
        tevs = list(map(lambda x: -TH_BETA*np.log(x['MS-GF:EValue']/TH_N0),
                    spec['SpectrumIdentificationItem']))
        empty_tevs = np.zeros(30)
        max_lim = min(30,len(tevs))
        empty_tevs[:max_lim] = sorted(tevs, reverse=True)[:max_lim]
        return empty_tevs, charge

    def __parse_msgf_mzid(self, input_path):
        """Extract MSGF+ scores (mzid)"""
        data = mzid.read(input_path)
        results = list(map(self.__get_msgf_mzid_tev, data))
        tevs = [x[0] for x in results]
        charges = [x[1] for x in results]
        return np.array(tevs), np.array(charges)


    @staticmethod
    def __find_mle_pars(scores):
        """estimate parameters using MLE"""

        length = len(scores[0,:])
        params = [None,]*(length-1)

        for hit in range(1, length): # skip first hit since it's a mixture
            cur_tev = scores[:, hit]
            cur_tev = cur_tev[cur_tev > 0.01] #dommad
            len_ = len(cur_tev)
            cur_tev = cur_tev[int(len_*0.01):int(len_*0.99)]
            params[hit-1] = ofs.mle_mubeta(cur_tev, hit)

        return list(zip(*params))


############### PARAMETER ESTIMATION ###################################

    def __get_mle_pars(self, tevs):
        """Get mu and beta from MLE + linear regression"""
        mu_, beta = self.__find_mle_pars(tevs)
        l_lim, u_lim = self.lr_cutoff
        trim_n0 = mu_[l_lim:u_lim]
        trim_a = beta[l_lim:u_lim]
        linreg = st.linregress(trim_n0, trim_a)

        return mu_, beta, linreg

    def __get_mm_pars(self, tev):
        """Get method of moments parameters"""
        m_1 = []
        m_2 = []

        for order in range(1,len(tev[0,:])):
            cur_scores = tev[:, order]
            cur_scores = cur_scores[cur_scores > 0]
            cur_m1, cur_m2 = ofs.mm_estimator(cur_scores, order)
            m_1.append(cur_m1)
            m_2.append(cur_m2)
        return m_1, m_2


################# BIC OPTIMIZATION ###################

    def bic_optimize(self, tev, **kwargs):
        """BIC optimization for linear regression mode parameters"""
        errors = []
        mu_range = np.linspace(0.1, 0.2, 500)

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

        # fig, axs = plt.subplots()
        # axs.plot(mu_range, errors)
        # axs.scatter(opt_mu, min(errors), color='orange')

        return opt_mu, opt_beta

   ############### VALIDATION with BH procedure #############

    def __parse_get_pvals(self, paths, pars, option='Tide'):
        """Parse the data and get lower-order and Coute's p-vals at the same time"""
        pvs = deque()
        ground_truth = deque()
        charges = deque()
        scores = deque()
        sidaks = deque()

        keywords = ["rand", "dec", "pos"]
        labels = [0, 4, 1] # 0: random, 4: decoy, 1: target

        big_data = list(map(fu.partial(self.__parse_data,
                                        keywords=keywords,
                                        paths=paths, labels=labels, pars=pars, option=option),
                                        np.arange(3)))

        for item in big_data:
            scores += item[0]
            charges += item[1]
            pvs += item[2]
            sidaks += item[3]
            ground_truth += item[4]

        return np.array(scores), np.array(charges), np.array(pvs), np.array(sidaks), np.array(ground_truth)


    def __get_val_data(self, row, pars):
        num_match = row['search_hit'][0]['num_matched_peptides']
        p_val = row['search_hit'][0]['search_score']['exact_pvalue']
        cur_tev = -TH_BETA*np.log(max(p_val, 10e-16)*num_match/TH_N0)
        sidak_pv = 1-pow(1-p_val, num_match)
        charge = int(row['assumed_charge'])

        if charge not in self.all_charges:
            fin_pv = 1
        else:
            fin_pv = 1 - st.gumbel_r.cdf(cur_tev, pars[charge][0], pars[charge][1])

        return cur_tev, charge, fin_pv, sidak_pv

    def __get_val_data_comet(self, row, pars):
        num_match = row['search_hit'][0]['num_matched_peptides']
        e_val = row['search_hit'][0]['search_score']['expect']
        p_val = e_val/num_match
        cur_tev = -TH_BETA*np.log(max(e_val, 10e-16)/TH_N0)
        sidak_pv = 1-pow(1-p_val, num_match)
        charge = int(row['assumed_charge'])

        if charge not in self.all_charges:
            fin_pv = 1
        else:
            fin_pv = 1 - st.gumbel_r.cdf(cur_tev, pars[charge][0], pars[charge][1])

        return cur_tev, charge, fin_pv, sidak_pv

   #process randoms
    def __parse_data(self, idx, keywords, paths, labels, pars, option='Tide'):

        keyword = keywords[idx]
        label_value = labels[idx]
        rand_paths = list(filter(lambda x: keyword in x, paths))
        items = deque()

        for pepxml_file in rand_paths:
            cur_file = pepxml.read(pepxml_file)
            if option == 'Comet':
                items += list(cur_file.map(self.__get_val_data_comet, args=(pars,)))
            else:
                items += list(cur_file.map(self.__get_val_data, args=(pars,)))

        scores = [x[0] for x in items]
        charges = [x[1] for x in items]
        pvs = [x[2] for x in items]
        sidaks = [x[3] for x in items]
        lbl = list(label_value*np.ones(len(items)))

        return scores, charges, pvs, sidaks, lbl
        #return data, labels

    #when FDR is calculated using BH method
    def __reps_single(self, reps, length, pvs, labels, len_correct):
        """Get FDP and TP for a single repetition"""
        random.seed()
        new_sel = random.choices(length, k=len(length))
        new_pvs = pvs[new_sel]
        new_labels = labels[new_sel]

        return self.fdr_lower(new_pvs, new_labels, len_correct)


    def __bootstrap_fdr(self, reps, labels, pvs, len_correct):
        """generate bootstrapped FDP estimates, get subsample from dataframe
            and calculate the stats for it, repeat"""

        length = np.arange(len(labels))
        #fdrs = np.linspace(0.0001, 0.1, 100)
        fdrs = np.zeros((reps, 100))
        fdps = np.zeros((reps, 100))
        tps = np.zeros((reps, 100))

        data = list(map(fu.partial(self.__reps_single, length=length,
                                            pvs=pvs, labels=labels,
                                            len_correct=len_correct), np.zeros(reps)))
        fdrs = [x[0] for x in data]
        fdps = [x[1] for x in data]
        tps = [x[2] for x in data]

        return fdrs, fdps, tps

    @staticmethod
    def __map_add_pvs(idx, scores, charges, pars):
        """calculate p-value for given score and set of params"""
        tev = scores[idx]
        charge  = charges[idx]
        pv_ = 1 - st.gumbel_r.cdf(tev, pars[charge][0], pars[charge][1])
        return pv_

    def __add_pvs(self, scores, charges, params):
        """Calculates p-values from TEV scores separated by charge"""
        indices = np.arange(len(scores))
        pvs = list(map(fu.partial(self.__map_add_pvs,
                                    scores=scores,
                                    charges=charges,
                                    pars=params), indices))
        return np.array(pvs)


    def add_peps(self, dfs, params, colname='pep_em'):
        """add PEPs from different EM variants to dataframe"""
        pvs = np.zeros(len(dfs))

        for pos, idx in enumerate(dfs.index):
            cur_tev = dfs.loc[idx, 'tev']
            charge = int(dfs.loc[idx, 'charge'])
            old_mu1, old_beta, old_mu2, old_sigma, old_pi0 = params[charge]

            if charge in self.all_charges:
                neg = ofs.pdf_mubeta(cur_tev, old_mu1, old_beta, 0)
                posit = st.norm.pdf(cur_tev, old_mu2, old_sigma)
                pep = old_pi0*neg/(old_pi0*neg + (1-old_pi0)*posit)
            else:
                pep = 1

            #to prevent inaccurate fit of positive model to mess up the results
            if cur_tev <=0.15:
                pep = 1

            pvs[pos] = pep

        dfs[colname] = pvs

        return dfs

    @staticmethod
    def pep_fdr(dfs, charge, colname):
        """Calculate FDR based on PEP values"""
        # colname is the name of PEP column
        dfs = dfs[(dfs["label"] != 4) & (dfs["charge"] == charge)]
        dfs.sort_values(colname, ascending=True, inplace=True)
        dfs.reset_index(inplace=True, drop=True)
        dfs.index += 1
        dfs['fdr'] = dfs[colname].cumsum()/dfs.index
        dfs['fdp'] = (dfs.index - dfs['label'].cumsum())/dfs.index
        dfs['tp'] = dfs['label'].cumsum()/len(dfs[dfs['label'] == 1])

        return dfs['fdr'].to_numpy(), dfs['fdr'].to_numpy(), dfs['tp'].to_numpy()


    @staticmethod
    def get_fdr(fdr, pvs, labels, len_correct, idx_for_bh):
        """Calculate FDR using BH procedure"""
        bh_ = idx_for_bh*fdr/len(pvs)
        adj_index = np.where(pvs <= bh_)[0]
        len_accepted = len(adj_index)
        adj_labels = labels[adj_index]

        if len_accepted == 0:
            len_accepted = 1

        if len_correct == 0:
            len_correct = 0

        len_tps = len(adj_labels[adj_labels == 1])
        fdp = 1-len_tps/len_accepted
        if fdp == 1:
            fdp = 0
        # dec = 2*len(ch3[ch3['label'] == 4])/len(ch3)
        # tps = len_tps/len_correct
        tps = len_tps # dommad: return only the number of TP PSMs

        return fdp, tps

    @staticmethod
    def __get_pi0(coute_pvs):
        """Get pi0 estimate using the graphical method describe in Coute et al."""
        compl_pvs = 1 - coute_pvs
        sorted_compls = np.sort(compl_pvs)
        dfs = pd.DataFrame(sorted_compls, columns=['score'])
        dfs.index += 1
        dfs['cdf'] = dfs.index/len(dfs)

        l_lim = int(0.4*len(dfs))
        u_lim = int(0.6*len(dfs))
        lr_ = st.linregress(dfs['score'][l_lim:u_lim], dfs['cdf'][l_lim:u_lim])
        return lr_.slope


    def fdr_lower(self, pvs, labels, len_correct):
        """Generate FDP and TP for the selected FDR range"""
        fdrs = np.linspace(0.001, 0.5, 1000)

        # select only target PSMs of the desired charge
        sorted_index = np.argsort(pvs)
        idx_for_bh = np.arange(len(pvs)) + 1
        sorted_pvs = pvs[sorted_index]
        sorted_labels = labels[sorted_index]

        #faster code for FDR calculation
        data = list(map(fu.partial(self.get_fdr, pvs=sorted_pvs,
                                   labels=sorted_labels,
                                   len_correct=len_correct,
                                   idx_for_bh=idx_for_bh), fdrs))

        fdps = [x[0] for x in data]
        tps = [x[1] for x in data]

        return fdrs, fdps, tps


    @staticmethod
    def plot_val_results(axs, fdrs, fdps, tps, lim, col1, col2):
        """Plot both FDP and TP vs FDR in the same plot"""
        #fig, ax = plt.subplots(figsize=(6,6))
        #lim = 0.1
        axs.grid(color='gray', linestyle='--', linewidth=1, alpha=0.2)
        axs.plot(fdrs, fdps, color=col1)
        axs.scatter(fdrs, fdps, marker='.', color=col1)
        axs.set_ylabel("FDP")
        ax2=axs.twinx()
        #ax.plot(fdr, decs)
        #ax.scatter(fdr, decs, marker='.')
        ax2.plot(fdrs, tps, marker='.', color=col2)
        ax2.set_ylabel("TPR", color=col2)
        ax2.set_ylim(0,1)
        axs.plot([0,lim], [0,lim], color='k', alpha=0.5)
        axs.set_xlim(-0.001, lim)
        axs.set_ylim(-0.001, lim)
        #ax2.legend(['lower', 'decoys', 'x-y'])

    @staticmethod
    def fdr_dec(dfs, charge):
        """calculation of FDP based on decoy counting"""
        fdp = []
        fdrs = []
        # decs = []
        # peps_low = []
        # peps_decs = []
        tps = []
        dfs = dfs[dfs.charge == charge]
        dfs.sort_values("tev", ascending=False, inplace=True)

        for i in np.linspace(1, len(dfs), 1000):
            if i == 0:
                continue
            data = dfs.iloc[:int(i), :]
            fdp_val = 1-len(data[data['label'] == 1])/len(data)
            dec = 2*len(data[data['label'] == 4])/len(data)
            tp_val = len(data[data.label == 1])/len(dfs[dfs.label==1])
            tps.append(tp_val)
            fdp.append(fdp_val)
            fdrs.append(dec)

        # fig, ax = plt.subplots(figsize=(6,6))
        # lim = 0.1
        # ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.2)
        # ax.plot(fdrs, fdp, color='royalblue')
        # ax.scatter(fdrs, fdp, marker='.', color='royalblue')
        # ax.set_ylabel("FDP", color='royalblue')
        # ax2=ax.twinx()
        # #ax.plot(fdr, decs)
        # #ax.scatter(fdr, decs, marker='.')
        # ax2.plot(fdrs, tps, color='orange')
        # ax2.set_ylabel("TPR", color='orange')
        # ax2.set_ylim(0,1)
        # ax.plot([0,lim], [0,lim], color='k', alpha=0.5)
        # ax.set_xlim(-0.001, lim)
        # ax.set_ylim(-0.001, lim)
        # #ax2.legend(['lower', 'decoys', 'x-y'])
        return fdrs, fdp, tps


class PiZeroEstimator:
    """estimate pi0 for given set of p-values"""

    def __init__(self):
        pass

    @staticmethod
    def get_pi0_b(pvs, b_val):
        """calculate pi0 estimate for given b value"""
        i = 1
        condition = False

        while condition is False:
            t_i = (i-1)/b_val
            t_iplus = i/b_val
            ns_i = len(pvs[(pvs < t_iplus) & (pvs >= t_i)])
            nb_i = len(pvs[pvs >= t_i])
            condition = bool(ns_i <= nb_i/(b_val - i + 1))
            i += 1

        i -= 1

        summand = 0
        for j in range(i-1, b_val+1):
            t_j = (j-1)/b_val
            summand += len(pvs[pvs >= t_j])/((1-t_j)*len(pvs))

        pi_0 = 1/(b_val - i + 2)*summand

        return pi_0

    def get_all_pi0s(self, pvs):
        """calculate pi0 for each b value"""
        # B is from I = {5, 10, 20, 50, 100}

        pi0s = []
        b_set = [5, 10, 20, 50, 100]

        for b_val in b_set:
            pi0s.append(self.get_pi0_b(pvs, b_val))
        
        return pi0s


    def get_boostrap_pi0s(self, pvs, no_reps, b_val):

        pi0_estimates = np.zeros(no_reps)

        for rep in range(no_reps):
            random.seed()
            new_pvs = np.array(random.choices(pvs, k=len(pvs)))
            pi0_estimates[rep] = self.get_pi0_b(new_pvs, b_val)

        return pi0_estimates

    @staticmethod
    def get_mse(pi0_bootstrap, pi0_true):
        """Calculates MSE for given set of p-values and true pi0 value"""
        summand = 0
        for i in range(len(pi0_bootstrap)):
            summand += pow(pi0_bootstrap[i] - pi0_true, 2)
        
        return summand/len(pi0_bootstrap)


    def find_optimal_pi0(self, pvs, n_reps):
        """Find the optimal pi0 according to Jiang and Doerge (2008)"""
        # compute pi0 for each B
        pi0_estimates = self.get_all_pi0s(pvs)
        pi0_ave = np.mean(pi0_estimates)
        b_set = [5, 10, 20, 50, 100]

        # compute MSE for each pi0 estimate
        mses = []

        for pi0_estim, b_val in zip(pi0_estimates, b_set):
            bootstraps = self.get_boostrap_pi0s(pvs, n_reps, b_val)
            mses.append(self.get_mse(bootstraps, pi0_ave))

        optimal_idx = mses.index(sorted(mses)[0])
        print(mses)
        print(pi0_estimates)
        print(optimal_idx)
        return pi0_estimates[optimal_idx]