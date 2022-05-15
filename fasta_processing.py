#Author: Dominik Madej (06.03.2019)

import pandas as pd
from Bio import SeqIO
import numpy as np
from random import shuffle
import sys

#remove non-decoys from the protein sequence database
def remove_nd(inputfile, alt_name = ''):
    fasta_sequences = SeqIO.parse(open(inputfile), 'fasta')

    if alt_name != '':
        text_file = open(alt_name, 'w')
    else:
        text_file = open('nd_' + inputfile, 'w')

    for fasta in fasta_sequences:
        descr = fasta.description
        seq = str(fasta.seq)
        el_list = descr.split('_')
        if el_list[0] == 'DECOY':
            text_file.write(">" + descr + '\n' + seq + '\n')
    text_file.close()

#generate list of peptides after protein digestion by trypsin
#credit: github.com/yafeng
def TRYPSIN(proseq, miss_cleavage):
    peptides = []
    cut_sites = [0]
    for i in range(0, len(proseq) - 1):
        if proseq[i] == 'K' and proseq[i + 1] != 'P':
            cut_sites.append(i + 1)
        elif proseq[i] == 'R' and proseq[i + 1] != 'P':
            cut_sites.append(i + 1)

    if cut_sites[-1] != len(proseq):
        cut_sites.append(len(proseq))

    if len(cut_sites) > 2:
        if miss_cleavage == 0:
            for j in range(0, len(cut_sites) - 1):
                peptides.append(proseq[cut_sites[j]:cut_sites[j + 1]])

        elif miss_cleavage == 1:
            for j in range(0, len(cut_sites) - 2):
                peptides.append(proseq[cut_sites[j]:cut_sites[j + 1]])
                peptides.append(proseq[cut_sites[j]:cut_sites[j + 2]])

            peptides.append(proseq[cut_sites[-2]:cut_sites[-1]])

        elif miss_cleavage == 2:
            for j in range(0, len(cut_sites) - 3):
                peptides.append(proseq[cut_sites[j]:cut_sites[j + 1]])
                peptides.append(proseq[cut_sites[j]:cut_sites[j + 2]])
                peptides.append(proseq[cut_sites[j]:cut_sites[j + 3]])

            peptides.append(proseq[cut_sites[-3]:cut_sites[-2]])
            peptides.append(proseq[cut_sites[-3]:cut_sites[-1]])
            peptides.append(proseq[cut_sites[-2]:cut_sites[-1]])
    else:  # there is no trypsin site in the protein sequence
        peptides.append(proseq)
    return peptides

#export the peptide sequences (length >= 7) after digestion to csv file
def digest_database(inputfile):
    handle = SeqIO.parse(inputfile, 'fasta')
    core = inputfile.split('.')[0]
    output = open(core + '.csv', 'w')

    for record in handle:
        proseq = str(record.seq)
        peptide_list = TRYPSIN(proseq, 0)
        for peptide in peptide_list:
            if len(peptide) >= 7:
                output.write(record.id + '\t' + peptide + '\n')

    handle.close()

#generate list of peptides (length >=7) shared by query and decoy databases
def gen_shared(query_csv, decoy_csv):
    query = pd.read_csv(query_csv.split('.')[0] + '.csv', delimiter='\t', header=None)
    query_unique = query.drop_duplicates([1])
    decoy = pd.read_csv(decoy_csv.split('.')[0] + '.csv', delimiter='\t', header=None)
    decoy_unique = decoy.drop_duplicates([1])
    shared = list(set.intersection(set(list(query_unique[1])), set(decoy_unique[1])))
    return shared

#replace shared peptides by shuffling the query organism decoy sequences
def replace_shared(decoy_input, list_shared):
    np.random.seed()
    handle = SeqIO.parse(decoy_input, 'fasta')
    output = open('no_shared_' + decoy_input, 'w')

    for record in handle:
        proseq = str(record.seq)
        descr = record.description

        for element in list_shared:

            if element in proseq:
                cut_sequence = proseq.split(element)
                split_element = list(element)
                shuffle(split_element)
                glued_element = ''.join(split_element)
                while glued_element == element:
                    np.random.seed()
                    shuffle(split_element)
                    glued_element = ''.join(split_element)
                #print(f'replacement successful? {glued_element != element}')
                new_proseq = [cut_sequence[0]]
                for i in range(1, len(cut_sequence)):
                    new_proseq.append(glued_element + cut_sequence[i])
                proseq = ''.join(new_proseq)
            if element not in proseq:
                continue
        output.write(">" + descr + '\n' + proseq + '\n')

    handle.close()

def remove_shared(decoy_input, list_shared):

    handle = SeqIO.parse(decoy_input, 'fasta')
    output = open('no_shared_' + decoy_input, 'w')

    for record in handle:
        proseq = str(record.seq)
        descr = record.description
        count = 0
        for element in list_shared:

            if element in proseq:
                count += 1
            if element not in proseq:
                continue
        if count == 0:
            output.write(">" + descr + '\n' + proseq + '\n')

    handle.close()



def main_work(query_t_fasta, decoy_t_fasta):

    # 1. remove non-decoys from the target-decoy database of the query
    #remove_nd(query_td_fasta)

    # 2. digest the decoy protein sequences into peptide sequences
    digest_database(query_t_fasta)
    digest_database(decoy_t_fasta)

    # 3. generate list of shared peptides between pure decoy and query target database
    shared = gen_shared(query_t_fasta, decoy_t_fasta)

    # 4. if no shared, finish; if there are some shared, reshuffle
    if shared == []:
        remove_nd(inputfile=decoy_t_fasta, alt_name='no_shared_' + decoy_t_fasta)
        print('Done!')
    else:
        remove_shared(decoy_t_fasta, shared)
        print('shared replaced, Done!')


def usage():
    print('Usage: \n --> to remove non-decoys: \n \
    first param: target-decoy database of the query organism (fasta),'
          '\n --> to extract real decoys: \n \
    first param: target-only database of the query organism (fasta),\n \
    second param: target-only database of the decoy organism (fasta)')

if __name__ == '__main__':

    if len(sys.argv) == 2:
        remove_nd(inputfile=sys.argv[1], alt_name='decoy_' + sys.argv[1])
    if len(sys.argv) == 3:
        main_work(sys.argv[1], sys.argv[2])
    if len(sys.argv) not in set([2,3]):
        usage()

