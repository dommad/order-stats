from pyteomics import mzxml
from pyteomics import pepxml
import numpy as np
from statistics import median
import pandas as pd
import deepdish as dd
import scipy.stats as st
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress as lr
import seaborn as sns
import pickle
import sys
from xml.etree import ElementTree as ET

def det_fdr_prophet(interactfile):
    p_vals = []
    d = pepxml.read(interactfile)
    print('reading pepxml...')
    for el in d:
        if 'search_hit' in el.keys():
            
            if 'DECOY' not in el['search_hit'][0]['proteins'][0]['protein']:

                p_v = el['search_hit'][0]['analysis_result'][0]['peptideprophet_result']['probability']
                spec = int(el['spectrum'].split('.')[-2])
                pep = el['search_hit'][0]['peptide']

                new_seq = pep.replace('I', 'X')
                new_seq = new_seq.replace('L', 'X')

                p_vals.append([p_v, spec, new_seq])

    df = pd.DataFrame(p_vals)
    df.columns = ['score', 'scan_id', 'peptide']

    df = df.sort_values('score', inplace=False, ascending=True)

    df = df.reset_index(drop=True)

    df.index += 1

    tree = ET.parse(interactfile)
    root = tree.getroot()
    #[34] for FDR 0.001
    #[45] for FDR 0.01
    pthreshold = float(root[0][0][1][45].attrib['min_prob'])
    print(f'pp threshold: {pthreshold}')
    finaldf = df[df['score'] >= pthreshold]

    final_dict = dict(zip(list(finaldf['scan_id']),list(finaldf['peptide'])))

    return final_dict


def refine(com_input, idpy_input, dict_name):

    com_dict = det_fdr_prophet(com_input)
    idpy_dict = det_fdr_prophet(idpy_input)
    core = com_input.split('.')[0]

    shared_ids = set.intersection(set(com_dict.keys()), set(idpy_dict.keys()))
    print(f'shared ids between comet and msfragger: {len(shared_ids)}')
    final_dict = {}

    for el in shared_ids:
        print(f' ID is: {el}')
        print(f'comet pep: {com_dict[el]}')
        print(f'msfragger pep: {idpy_dict[el]}')
        if com_dict[el] == idpy_dict[el]:

            final_dict[el] = com_dict[el]

    print(f'length of dictionary: {len(final_dict.keys())}')
    pickle.dump(final_dict, open(f'{dict_name}.pkl', 'wb'))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('check the parameters!: PP comet input, PP idpy input, name of dictionary (without extension), synthetic csv, id_no')
    if len(sys.argv) == 4:
        refine(sys.argv[1], sys.argv[2], sys.argv[3])
