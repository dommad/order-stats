import random
from collections import deque
import scipy.stats as st
import numpy as np
import pandas as pd
import orderstats.stat as of
import parsers

class Validation:

    def __init__(self, outname, bic_cutoff=0.17, lr_cutoff=(5,10)):
        self.params_est = []
        self.out = outname
        self.len_correct = 0
        self.bic_cutoff = bic_cutoff
        self.tevs = []
        self.lr_cutoff = lr_cutoff
        self.all_charges = [2,3,4]


    def parse_decoy_data(self):
        pass

    def parse_all_files(self, input_files, engine):

        all_dfs = []

        try:
            parser_instance = getattr(parsers, f"{engine}Parser")()
        except AttributeError:
            raise ValueError(f"Unsupported or invalid engine: {engine}")
        
        parser_instance = getattr(parsers, f"{engine}Parser")()

        for file in input_files:
            psms_df = parser_instance.parse(file)
            all_dfs.append(psms_df)
        
        master_df = pd.concat(all_dfs, axis=0, ignore_index=True)

        return master_df

    def add_new_pvs(self, params):
        pass

    def new_bootstrap(self, df):
        pass


    def execute_validation(self, input_files: list, dec_paths: list, bootstrap_reps=500, ext_params: dict = {}, mode: str = 'Tide', plot=False):
        """read the pepxml, automatically add the p-values based on lower order estimates"""

        if ext_params:
            self.params_est = ext_params

        # TODO: support all search engines and format, not just Comet and Tide, eliminate "mode"
        # this parsing function must be able to process target-only, randomized, decoy-only files 
        # and produce tev scores, charges, lower_order_pvalues, coute (Sidak) p_values, ground_truth_labels, i.e., labels for target, decoy, random
        #tev_scores, charges, lower_order_pvalues, coute_pvs, labels = self.__parse_get_pvals(input_files, self.params_est, option=mode)
        master_df = self.parse_all_files()
        # master_df.columsn = ['tev', 'charge', 'hit_rank', 'lower_order_pv', 'coute(sidak)_pv', 'label', ...]

        # TODO: fix the parsing thing
        # return tevs, np.array(charges)
        # new return will be df[['tev', 'charge', 'hit_rank', ...]]
        decoy_df = self.parse_decoy_data()

        decoy_params = self.get_top_decoy_parameters(decoy_df)
        cdd_params = pd.read_csv('./cdd_params.txt', header=None).to_numpy()

        pi_0 = len(lower_order_pvalues[np.where(labels == 0)])/len(lower_order_pvalues[np.where(labels != 4)])

        # alternatively, pi_0 can be estimated as outlined in Jiang and Doerge (2008):
        # pi_0 = PiZeroEstimator().find_optimal_pi0(low_pvs, 10)


        # this should be replaced by a function adding p-values, can be either decoy or cdd-based
        # decoy_pvalues = self.__add_pvs(tev_scores, charges, decoy_params)
        # cdd_pvalues = self.__add_pvs(tev_scores, charges, cdd_params)
        master_df['decoy_pv'] = self.new_add_pvs(decoy_params)
        master_df['cdd_pv'] = self.new_add_pvs(cdd_params)


        no_decoys_df = master_df[master_df['label'] != 4].copy()


        # idxs_non_decoys = list(set(np.where(labels != 4)[0]))
        # cur_labels = labels[idxs_non_decoys]
        # cur_lower = lower_order_pvalues[idxs_non_decoys]
        # cur_decoy = decoy_pvalues[idxs_non_decoys]
        # cur_cdd = cdd_pvalues[idxs_non_decoys]
        # cur_coute = coute_pvs[idxs_non_decoys]

        self.len_correct = len(cur_labels[cur_labels == 1])

        #TODO: refactor to process all types of p-values in one go in bootstrap function
        # stats_low = self.__bootstrap(bootstrap_reps, cur_labels, cur_lower)
        # stats_dec = self.__bootstrap(bootstrap_reps, cur_labels, cur_decoy)
        # stats_cdd = self.__bootstrap(bootstrap_reps, cur_labels, cur_cdd)
        # stats_coute = self.__bootstrap(bootstrap_reps, cur_labels, cur_coute)

        # all_boot_stats.append([stats_low, stats_dec, stats_cdd, stats_coute])

        all_boot_stats = self.new_bootstrap(bootstrap_reps, no_decoys_df)

        if plot:
            self.plot_validation_results(all_boot_stats, pi_0)
        
        return all_boot_stats


    def plot_validation_results(self, all_boot_stats, pi_0):
        """Plot validation results"""

        fig, axs = plt.subplots(1, 2, figsize=(6,3))
        self.__plot_boot_fdrs(axs[0], all_boot_stats, pi_0)
        self.__plot_boot_tps(axs[1], all_boot_stats)
        fig.tight_layout()

        fig.savefig(f"./graphs/{self.out}_validation.png", dpi=600, bbox_inches='tight')



    def get_top_decoy_parameters(self, decoy_df):
        """Generate parameters of null top models by fitting Gumbel to TEVs of top-scoring decoys"""

        available_charges = set(decoy_df['charge'])
        top_psms = decoy_df[decoy_df['hit_rank'] == 1] 
        params = {}

        for charge in available_charges:
            cur_tevs = top_psms[top_psms['charge'] == charge]['tev']

            if len(cur_tevs) < 100:
                params[charge] = (0, 0)
                continue

            params[charge] = st.gumbel_r.fit(cur_tevs)
    
        return params


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

  ############### VALIDATION with BH procedure #############

    def __parse_get_pvals(self, paths, pars, option='Tide'):
        """Parse the data and get lower-order and Coute's p-vals at the same time"""
        pvs = deque()
        ground_truth_labels = deque()
        charges = deque()
        scores = deque()
        sidaks = deque()

        keywords = ["rand", "dec", "pos"]
        labels = [0, 4, 1] # 0: random (negative), 4: decoy, 1: target (positive)

        big_data = list(map(fu.partial(self.__parse_data,
                                        keywords=keywords,
                                        paths=paths, labels=labels, pars=pars, option=option),
                                        np.arange(3)))

        for item in big_data:
            scores += item[0]
            charges += item[1]
            pvs += item[2]
            sidaks += item[3]
            ground_truth_labels += item[4]

        return np.array(scores), np.array(charges), np.array(pvs), np.array(sidaks), np.array(ground_truth_labels)


    def __get_val_data(self, row, pars):

        num_match = row['search_hit'][0]['num_matched_peptides']
        p_val = row['search_hit'][0]['search_score']['exact_pvalue']
        cur_tev = -TH_BETA * np.log(max(p_val, 10e-16) * num_match / TH_N0)
        sidak_pv = 1 - pow(1 - p_val, num_match)
        charge = int(row['assumed_charge'])

        if charge not in self.all_charges:
            fin_pv = 1
        else:
            fin_pv = 1 - st.gumbel_r.cdf(cur_tev, pars[charge][0], pars[charge][1])

        return cur_tev, charge, fin_pv, sidak_pv


    def __get_val_data_comet(self, row, pars):
        num_match = row['search_hit'][0]['num_matched_peptides']
        e_val = row['search_hit'][0]['search_score']['expect']
        p_val = e_val / num_match
        cur_tev = -TH_BETA * np.log(max(e_val, 10e-16) / TH_N0)
        sidak_pv = 1 - pow(1 - p_val, num_match)
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
        # TODO: adding support for PeptideProphet results with CDD or for PEPs obtained from my EM algorithm
        pvs = np.zeros(len(dfs))

        for pos, idx in enumerate(dfs.index):
            cur_tev = dfs.loc[idx, 'tev']
            charge = int(dfs.loc[idx, 'charge'])
            old_mu1, old_beta, old_mu2, old_sigma, old_pi0 = params[charge]

            if charge in self.all_charges:
                neg = of.TEVDistribution().pdf(cur_tev, old_mu1, old_beta, 0)
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
        # TODO: add support for PEP-based FDR

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
