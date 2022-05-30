from pyteomics import pepxml
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET
import pickle
import sys
import yaml
import mpmath
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import time


def gumbel_pdf(x, m, beta):
    x, m, beta = mpmath.mpf(x), mpmath.mpf(m), mpmath.mpf(beta)
    if x < 0: return 0
    return (1 / beta) * (mpmath.exp(-(x + mpmath.exp(-x))))


def gumbel_cdf(x, m, beta):
    x, m, beta = mpmath.mpf(x), mpmath.mpf(m), mpmath.mpf(beta)
    return mpmath.exp(-mpmath.exp(-(x - m) / beta))


def det_fdr_prophet(interactfile, ref_p, fdr, no_files):
    
    d = pepxml.read(interactfile)
    
    pps = -1*np.ones(len(d))
    fvals = np.zeros(len(d))
    labels = np.zeros(len(d))
    charges = np.zeros(len(d))
    seqs = []
    specs = []
                    
    
    k=0
    for el in d:
       # if 'DECOY' in el['search_hit'][0]['proteins'][0]['protein']:
       #     continue
        if 'search_hit' in el.keys():
            p_v = el['search_hit'][0]['analysis_result'][0]['peptideprophet_result']['probability']
            spec = el['spectrum']
            ch = int(el['assumed_charge'])
            pep = el['search_hit'][0]['peptide']
            fval = el['search_hit'][0]['search_score']['expect']
            fval = -0.02 * np.log(fval / 1000.)

            new_seq = pep.replace('I', 'X').replace('L', 'X')
            scanid = int(spec.split('.')[-2])

            if scanid in ref_p.keys():
                if ref_p[scanid] == new_seq:
                    label = 1
                else:
                    label = 0
            if scanid not in ref_p.keys():
                label = 2
                
            pps[k] = p_v
            fvals[k] = fval
            labels[k] = label
            charges[k] = ch
            seqs.append(new_seq)
            specs.append(spec)
            
            k +=1
            
    

    df = pd.DataFrame(np.array([pps, fvals, labels]).T)
    df.columns = ['PP_pval', 'TEV', 'label']
    df = df[df['PP_pval'] != -1]
    df['peptide'] = seqs
    df['spectrum'] = specs
    df['charges'] = charges
    df = df[df['charges'] == 3]

    df = df.sort_values('PP_pval', inplace=False, ascending=True)
    df = df.reset_index(drop=True)
    df.index += 1

    tree = ET.parse(interactfile)
    root = tree.getroot()

    #34 0.001
    #45 0.01
    #47 0.02
    #50 0.04
    #51 0.05
    #49 0.03

    if fdr == 0.001:
        fdr_index = 34
    if fdr == 0.005:
        fdr_index = 40
    if fdr == 0.01:
        fdr_index = 45
    if fdr == 0.015:
        fdr_index = 46
    if fdr == 0.02:
        fdr_index = 47
    if fdr == 0.025:
        fdr_index = 48
    if fdr == 0.03:
        fdr_index = 49
    if fdr == 0.04:
        fdr_index = 50
    if fdr == 0.05:
        fdr_index = 51
		
    pthreshold = float(root[0][0][int(no_files)][fdr_index].attrib['min_prob'])
    print(f'pp threshold: {pthreshold}')
    finaldf = df[df['PP_pval'] >= pthreshold]

    return finaldf


#to be modified, rewrite indexing of dataframes from iat to at so that it's easier to understand columns
def q_val(df, fdr):
    q_vals = list((len(df) * df['PP_pval']) / df.index.to_series())
    df['q_val'] = pd.Series(q_vals)
    adj_q = []
    cur_th = 0
    for i in range(len(df.index) - 1, -1, -1):
        cur_q = df.iat[i, 4]
        if i == len(df.index):
            cur_th = cur_q
            adj_q.append(cur_th)
        else:
            if cur_q > cur_th:
                adj_q.append(cur_th)
            else:
                cur_th = cur_q
                adj_q.append(cur_th)
    adj_q.reverse()
    df['adj_q'] = pd.Series(adj_q)
    finaldf = df[df['adj_q'] <= fdr]
    return finaldf


def BH(df, fdr):
    BH = pd.Series((df.index.to_series() * fdr) / len(df))
    df['bh'] = BH
    finaldf = df[df['PP_pval'] <= df['bh']]
    return finaldf


def get_perc(poutxmlfile, fdr):
    tree = ET.parse(poutxmlfile)
    root = tree.getroot()

    psms = []
    q_vals = []
    pep_seq = []

    for psm in root[1]:
        psms.append(list(psm.attrib.values())[0])
        q_vals.append(float(psm[1].text))
        pep = psm[5].attrib['seq']
        clean_seq = ""
        for i in pep:
            if i.isalpha() == True:
                clean_seq += i
            else:
                continue

        pep = clean_seq
        # clean_seq = pep.replace('*', "")
        pep = pep.replace('I', "X").replace('L', 'X')
        pep_seq.append(pep)

    df = pd.DataFrame(np.array([psms, q_vals, pep_seq]).T)

    df.columns = ['spectrum', 'q_val', 'peptide']
    df['q_val'] = df['q_val'].astype(dtype='float')

    df = df.sort_values('q_val', inplace=False, ascending=True)
    finaldf = df[df['q_val'] <= fdr]
   
    return finaldf


def get_sh_spectra(df, ref_p, option='default'):
    shared = 0


    if option == 'default':

        for i in df.index:

            spec = df.loc[i,'spectrum']
            peptide = df.loc[i,'peptide']

            ind = int(spec.split('.')[-2])
            if ind in ref_p.keys():
                if ref_p[ind] == peptide:
                    shared += 1


    if option == 'percolator':

        for i in df.index:

            spec = df.loc[i,'spectrum']
            peptide = df.loc[i,'peptide']

            el = spec.split('_')
            # for concatenated file use int(el[-3]) - 1
            ind = int(el[-3])
            if ind in ref_p.keys():
                if ref_p[ind] == peptide:
                    shared += 1
             
    """
    if option == 'scan':
        for i in dictionary['scan']:
            ind = int(i)
            if ind in ids:
                shared += 1
                """

    return shared


def get_x_seq(df):
    perc_peps = []

    for k in df.index:
        pep = df.at[k, 'pep_seq']
        pep = list(pep)

        clean_seq = ""
        for i in pep:
            if i.isalpha() == True:
                clean_seq += i
            else:
                continue

        pep = clean_seq
        pep = pep.replace('I', "X").replace('L', 'X')
        perc_peps.append(pep)

    return perc_peps



#
def get_stats(sh_peps, sh_spectra, no_spectra, all_peps):


    no_sh_peps = len(sh_peps)
    real_fdr = 1 - sh_spectra / no_spectra
    p_sh_spectra = sh_spectra/no_spectra
    p_sh_peps = no_sh_peps / len(all_peps)

    total_stats = [str(sh_spectra), str(no_sh_peps), str(p_sh_spectra), str(p_sh_peps), str(real_fdr)]
    return total_stats



def cdd_charges(pepxmlfile, pars,fdr, ref_p):
    # slices is a list of lists: [[p1ch2, m], [p2ch2, m], ...]
    p_vals = []

    p1ch2 = pars['p1ch2']
    p2ch2 = pars['p2ch2']
    p1ch3 = pars['p1ch3']
    p2ch3 = pars['p2ch3']
    p1ch4 = pars['p1ch4']
    p2ch4 = pars['p2ch4']

    d = pepxml.read(pepxmlfile)
    specs = []
    seqs = []
    #IDs = np.zeros(len(d))
    pvs = -1*np.ones(len(d))
    labels = np.zeros(len(d))
    k=0
    for el in d:
        if 'search_hit' in el.keys():

            spec = el['spectrum']
            scanid = int(spec.split('.')[-2])
            f_val = -0.02 * np.log((el['search_hit'][0]['search_score']['expect']) / 500)
            charge = int(el['assumed_charge'])

            if charge == 2:
                p_v = 1 - gumbel_cdf(f_val, p1ch2, p2ch2)

            elif charge == 3:
                p_v = 1 - gumbel_cdf(f_val, p1ch3, p2ch3)

            else:
                p_v = 1 - gumbel_cdf(f_val, p1ch4, p2ch4)

            pep = el['search_hit'][0]['peptide']
            new_seq = pep.replace('I', 'X').replace('L', 'X')
            label = 0
            
            if scanid in ref_p.keys():
                if ref_p[scanid] == new_seq:
                    label = 1
                else:
                    label = 0
            if scanid not in ref_p.keys():
                label = 2
                
            specs.append(spec)
            seqs.append(new_seq)
            pvs[k] = p_v
            labels[k] = label
            k += 1
            
    df = pd.DataFrame(np.array([pvs, labels]).T)
    df.columns = ['PP_pval', 'label']
    df['spectrum'] = specs
    df['peptide'] = seqs
    df = df[df['PP_pval'] != -1]

    finaldf = BH(df, fdr)

    return finaldf


def get_results(filename, ref_p, fdr, no_files, mode, tp_peps, cdd_mode='integrated', distparams=""):
    
    ttt = time.time()
    if mode == 'PP':
        df = det_fdr_prophet(filename, ref_p, fdr, no_files)
        
    if mode == 'CDD':
        if cdd_mode == 'integrated':
            df = det_fdr_prophet(filename, ref_p, fdr, no_files)
        if cdd_mode == 'alone':
            pars = yaml.load(open(distparams,'rb'))
            df = cdd_charges(filename, pars, fdr, ref_p)
        
    if mode == 'Per':
        df = get_perc(filename, fdr)
        
    
    if mode == 'Per':
        TP_peps = set.intersection(set(tp_peps), set(list(df['peptide'])))
        TP_spec = get_sh_spectra(df, ref_p, option='percolator')
    else:
        TP_peps = set.intersection(set(tp_peps), set(list(df['peptide'])))
        TP_spec = get_sh_spectra(df, ref_p, option='default')
        
    real_FDP = 1 - TP_spec/len(df.index)
    all_stats = [real_FDP, len(TP_peps), TP_spec]
    df.to_csv(f'{mode}_test.csv')
    
    return all_stats


def ex_validation(synthetic, interact_td, interact_t, poutxmlfile, fdr, csvfile, no_files, distparams="", cdd_mode='integrated', notes=""):
    
    print(f'starting {interact_td}')
    mpmath.mp.dps = 50
    fdr = float(fdr)
    ref_p = pickle.load(open(synthetic, 'rb'))
    notes = str(notes)


    outname = interact_td.split('.')[0]
    tp_peps = list(set(list(ref_p.values())))
    ids = list(ref_p.keys())
    
    
    PP_stats = get_results(interact_td, ref_p, fdr, no_files, 'PP', tp_peps)
    CDD_stats = get_results(interact_t, ref_p, fdr, no_files, 'CDD', tp_peps, cdd_mode, distparams)
    Per_stats = get_results(poutxmlfile, ref_p, fdr, no_files, 'Per', tp_peps)
    
    fin = pd.Series([outname] + ['PP'] + PP_stats + ['CDD'] + CDD_stats + ['Per'] + Per_stats + [fdr])
    fin = fin.T
    
    current = pd.DataFrame()
    current = current.append(fin, ignore_index=True)
    current.to_csv(csvfile, mode='a', header=None, index=None)
                     
                  
def ex_BH(synthetic, interact_td, interact_t, fdr, csvfile, no_files, distparams=""):
    
    print(f'starting {interact_t}')
    mpmath.mp.dps = 50
    fdr = float(fdr)
    ref_p = pickle.load(open(synthetic, 'rb'))
   


    outname = interact_t.split('.')[0]
    tp_peps = list(set(list(ref_p.values())))
    ids = list(ref_p.keys())
    
    
    PP_stats = get_results(interact_td, ref_p, fdr, no_files, 'PP', tp_peps)
    BH_stats = get_results(interact_t, ref_p, fdr, no_files, 'CDD', tp_peps, cdd_mode='alone', distparams=distparams)
                     
    fin = pd.Series([outname] + ['PP'] + PP_stats + ['BH'] + BH_stats + [fdr])
    #fin = fin.T
    
    current = pd.DataFrame()
    current = current.append(fin, ignore_index=True)
    current.to_csv(csvfile, mode='a', header=None, index=None)         
    
    
def optimize_CDD(synthetic, interact_t, csvfile, fdr=0.01, no_files=1, distparams=""):
    
    print(f'starting {interact_t}')
    mpmath.mp.dps = 50
    fdr = float(fdr)
    ref_p = pickle.load(open(synthetic, 'rb'))
    mode = interact_t.split(".")[0].split("_")[-1]
   


    outname = interact_t.split('.')[0]
    tp_peps = list(set(list(ref_p.values())))
    ids = list(ref_p.keys())
    
    
    PP_stats = get_results(interact_t, ref_p, fdr, no_files, 'CDD', tp_peps, cdd_mode='integrated' )
                     
    fin = pd.Series([outname] + [mode] + PP_stats + [fdr])
    fin = fin.T
    
    current = pd.DataFrame()
    current = current.append(fin, ignore_index=True)
    current.to_csv(csvfile, mode='a', header=None, index=None)         
    
    
def optimize_PP(synthetic, interact_td, csvfile, fdr=0.01, no_files=1, distparams=""):
    
    print(f'starting {interact_td}')
    mpmath.mp.dps = 50
    fdr = float(fdr)
    ref_p = pickle.load(open(synthetic, 'rb'))
    mode = interact_td.split(".")[0].split("_")[-1]
   


    outname = interact_td.split('.')[0]
    tp_peps = list(set(list(ref_p.values())))
    ids = list(ref_p.keys())
    
    
    PP_stats = get_results(interact_td, ref_p, fdr, no_files, 'PP', tp_peps)
                     
    fin = pd.Series([outname] + [mode] + PP_stats + [fdr])
    fin = fin.T
    
    current = pd.DataFrame()
    current = current.append(fin, ignore_index=True)
    current.to_csv(csvfile, mode='a', header=None, index=None)         
    
        
        
        
if __name__ == '__main__':

    if len(sys.argv) != 11:
        print('synthetic, interact_td, interact_t, poutxmlfile, fdr, csvfile, distparams, cddmode (op), note (op)')
    if len(sys.argv) == 11:
        ex_validation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
    if len(sys.argv) == 8:
        ex_BH(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    if len(sys.argv) == 4:
        #optimize_CDD(sys.argv[1], sys.argv[2], sys.argv[3])
        optimize_PP(sys.argv[1], sys.argv[2], sys.argv[3])
            
