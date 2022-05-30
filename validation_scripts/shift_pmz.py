from pyteomics import pepxml
from pyopenms import *
import numpy as np
import pandas as pd
import scipy.stats as st
from xml.etree import ElementTree as ET
import pickle
import sys
import time
import deepdish as dd



def correct_PSMs(pepxmlfile, final_list):


    b = pepxml.read(pepxmlfile)
    correct_dict = {}

    for spectrum in b:
        if 'search_hit' in spectrum.keys():
            spec = str(spectrum['spectrum'])
            pep = str(spectrum['search_hit'][0]['peptide'])

            if len(pep) >= 7:
                new_seq = pep.replace('I', 'X')
                new_seq = new_seq.replace('L', 'X')

                if new_seq in final_list:
                    correct_dict[spec] = new_seq

    return correct_dict





def getMzML(mzml_file, GT):

    #before it was 30.8 and shift*cur_ch
    #now its 31 and shift/cur_ch
    shift = 31

    GT = pickle.load(open(GT, 'rb'))
    core = mzml_file.split('.')[0]
    mzml_dict = MSExperiment()
    if mzml_file.split('.')[-1]  == 'mzML':
        MzMLFile().load(mzml_file, mzml_dict)
    if mzml_file.split('.')[-1]  == 'mzXML':
        MzXMLFile().load(mzml_file, mzml_dict)

    new_mzml = MSExperiment()

    for i in range(len(mzml_dict.getSpectra())):
        if (i % 1000 == 0): print(f'{i}...', end='', flush=True)

        spec = mzml_dict[i]
        mslevel = spec.getMSLevel()

        if mslevel == 2:

            id = int(str(spec.getNativeID()).split('=')[-1])
            spectrum = MSSpectrum()
            peaks = spec.get_peaks()
            mz = peaks[0]
            i = peaks[1]
            spectrum.set_peaks([mz, i])

            spectrum.setDriftTime(spec.getDriftTime()) # 25 ms
            spectrum.setRT(spec.getRT()) # 205.2 s
            spectrum.setMSLevel(spec.getMSLevel()) # MS3
            spectrum.setNativeID(spec.getNativeID())

            prec = spec.getPrecursors()[0]
            p = Precursor()
            p.setIsolationWindowLowerOffset(prec.getIsolationWindowLowerOffset())
            p.setIsolationWindowUpperOffset(prec.getIsolationWindowUpperOffset())
            cur_ch = int(prec.getCharge())
    
            
            if id in GT.keys():
                p.setMZ(float(prec.getMZ())) # isolation at 600 +/- 1.5 Th
                p.setActivationEnergy(35) # 40 eV
                p.setCharge(int(prec.getCharge())) # 4+ ion
                spectrum.setPrecursors([p])
                new_mzml.addSpectrum(spectrum)
            else:
                #originally +19.5
                #now 30.8
                p.setMZ(float(prec.getMZ())+shift/cur_ch) # isolation at 600 +/- 1.5 Th
                p.setActivationEnergy(35) # 40 eV

                p.setCharge(prec.getCharge()) 

                spectrum.setPrecursors([p])
                new_mzml.addSpectrum(spectrum)

    if mzml_file.split('.')[-1] == 'mzML':
        MzMLFile().store(f'refined_' + core + '.mzML', new_mzml)
    if mzml_file.split('.')[-1] == 'mzXML':
        MzXMLFile().store(f'refined_' + core + '.mzXML', new_mzml)
    print('\n')


def organism(cons_dict, mzml_file):

    getMzML(mzml_file, cons_dict)


def usage():
    print('Usage: pool peps, file id, pepxml target, mzml file for human; consensus dict, mzml file for yeast"')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
    if len(sys.argv) == 3:
        #for yeast, instead of pool peps provide consensus dictionary (generated by refine_idpy_comet.py)
        organism(sys.argv[1], sys.argv[2])
