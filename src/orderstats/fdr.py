"""Methods used to estimate false discovery rate"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np



class FDRCalculator(ABC):

    @abstractmethod
    def calculate_fdp_tpr(self):
        pass


class BenjaminiHochberg(FDRCalculator):

    @staticmethod
    def calculate_fdp_tpr(df_sorted: pd.DataFrame, p_value_column: str, critical_array: np.ndarray, pos_label: str, neg_label: str):

        df_labels = df_sorted['gt_label'].to_numpy()
        sorted_pvals = df_sorted[p_value_column].to_numpy()
        len_df_correct_labels = len(df_labels[df_labels == pos_label])
        masks = [sorted_pvals <= x for x in critical_array]
        bh_gt_labels = [df_labels[mask] for mask in masks]

        fdps = [len(x[x == neg_label]) / len(x) for x in bh_gt_labels]
        tprs = [len(x[x == pos_label]) / len_df_correct_labels for x in bh_gt_labels]

        return fdps, tprs


class PosteriorErrorProb(FDRCalculator):
    pass


class DecoyCount(FDRCalculator):
    
    def calculate_fdr(self, df, score_name, decoy_factor):
        
        df.sort_values(score_name, ascending=False, inplace=True)
        df[f"{score_name}_cum_dec"] = decoy_factor * df["is_decoy"].cumsum() / (~df['is_decoy']).cumsum()

        df_no_decoys = df[~df["is_decoy"]].copy()
        no_decoys_index = np.arange(1, len(df_no_decoys)+1)
        df_no_decoys[f"{score_name}_cum_neg"] = (~df_no_decoys["gt_status"]).cumsum() / no_decoys_index

        fdr_dec = df_no_decoys["cum_dec"].to_numpy()
        fdp = df_no_decoys["cum_neg"].to_numpy()

        return (fdr_dec, fdp)



    @staticmethod
    def calculate_fdr_decoy_counting(dfs, charge):
        """calculation of FDP based on decoy counting"""
        fdp = []
        fdrs = []
 
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


        return fdrs, fdp, tps
    

    @staticmethod
    def calculate_fdr_bh(fdr, pvs, labels, len_correct, idx_for_bh):
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
    def calculate_fdr_posterior_error_prob(dfs, charge, colname):
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