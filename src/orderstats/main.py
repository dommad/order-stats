"""Full analysis of pepxml file using lower order statistics"""
from utils import *
import src.orderstats.stat as of
import parsers
from plot import Plotting




class Parsers:

    def __init__(self) -> None:
        pass
        # TODO: implement parser for MSGF and refactor all codes
        # so that these functions can be replaced by the general parsers

 ########## PARSING TIDE AND COMET ####################

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
        # TODO: try not to use pyteomics and have my own mzid parser based on xml
        data = mzid.read(input_path)
        results = list(map(self.__get_msgf_mzid_tev, data))
        tevs = [x[0] for x in results]
        charges = [x[1] for x in results]
        return np.array(tevs), np.array(charges)


