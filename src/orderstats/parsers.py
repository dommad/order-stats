"""Module for processing data from spectral libraries in format .sptxt"""
import re
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import csv

FILE_FORMATS = ['pep.xml', 'txt', 'mzid']
SEARCH_ENGINES = ['Comet', 'SpectraST', 'Tide', 'MSFRagger', 'MSGF+']


class ProteomicsDataParser(ABC):
    def __init__(self):
        self.engine = ""


    def parse(self, file_name):
        after_dots = file_name.lower().split('/')[-1].split('.')
        
        if after_dots[-2:] == ['pep', 'xml']:
            file_ext = 'pepxml'
        elif after_dots[-1] in FILE_FORMATS:
            file_ext = after_dots[-1]
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")

        return getattr(self, f"parse_{file_ext}")(file_name)


    @abstractmethod
    def update_headers_pepxml(self, headers):
        pass

    @abstractmethod
    def add_modification_data_pepxml(self, spectrum_query, ns):
        modification_list = spectrum_query.findall('.//pepXML:modification_info', namespaces=ns)
        
        if modification_list:
            modification_info = [modification_list[0].attrib['modified_peptide']]
        else:
            modification_info = [None]
        return modification_info


    def parse_pepxml(self, file_name):
        """Parses pepxml (Comet) to tsv and outputs pandas dataframe"""
        try:
            core_name = file_name.split('/')[-1].split('.')[0]
            tree = ET.parse(file_name)
            root = tree.getroot()

            # Define the namespace used in the XML
            ns = {'pepXML': 'http://regis-web.systemsbiology.net/pepXML'}

            with open(f"{core_name}_{self.engine}.tsv", 'w', newline='') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')

                # Extract headers from the first spectrum_query element
                first_spectrum_query = root.findall('.//pepXML:spectrum_query', namespaces=ns)[0]

                if first_spectrum_query:
                    headers = self.extract_headers_pepxml(first_spectrum_query, ns)
                    writer.writerow(self.update_headers_pepxml(headers))
                else:
                    print("No data to process!")
                    return

                for spectrum_query in root.findall('.//pepXML:spectrum_query', namespaces=ns):
                    spectrum_info = self.extract_values_to_list_pepxml(spectrum_query.items())
                    search_hit_list = spectrum_query.findall('.//pepXML:search_hit', namespaces=ns)
                    
                    modification_info = self.add_modification_data_pepxml(spectrum_query, ns)
                 
                    for search_hit in search_hit_list:
                        search_hit_info = self.extract_values_to_list_pepxml(search_hit.items())
                        search_score_list = search_hit.findall('.//pepXML:search_score', namespaces=ns)
                        search_score_info = [score.get('value') for score in search_score_list]

                        combined = spectrum_info + search_hit_info + search_score_info + modification_info
                        writer.writerow(combined)

            # TODO: is this step necessary?
            df = pd.read_csv(f"{core_name}_{self.engine}.tsv", sep='\t')
            df.rename(columns=self.rename_columns(), inplace=True)
            # TODO: this part may be different for different engines, the row may have a differnt name,
            # so we need to export this login to individual engine parsers
            df['is_decoy'] = [True if 'DECOY' in row.protein else False for row in df.itertuples()]

            return df

        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


    def parse_txt(self, file_name, sep='\t'):
        # Implement common parsing logic
        df = pd.read_csv(file_name, sep=sep)
        # renaming will be handled by individual engines
        df.rename(columns=self.rename_columns(), inplace=True)
        return df


    def parse_mzid(self):
        # Implement common parsing logic
        pass


    def rename_columns(self):
        pass


    def extract_headers_pepxml(self, element, ns):
        """Extracting header names from the first spectrum_query in pepxml"""

        extract_names_to_list = lambda list_tuples: [x[0] for x in list_tuples]
        headers = extract_names_to_list(element.items())

        # Extract headers from the first search_hit element in the spectrum_query
        first_search_hit = element.findall('.//pepXML:search_hit', namespaces=ns)[0]
        if first_search_hit:
            headers += extract_names_to_list(first_search_hit.items())

            # Extract search_score keys and add to headers
            search_score_list = first_search_hit.findall('.//pepXML:search_score', namespaces=ns)
            search_score_keys = [score.get('name') for score in search_score_list]
            headers += search_score_keys

        return headers


    def extract_values_to_list_pepxml(self, list_tuples):
        return [x[1] for x in list_tuples]



class CometParser(ProteomicsDataParser):
    def __init__(self):
        super().__init__()
        self.engine = "Comet"

    def rename_columns(self):
        # Define column renaming logic specific to Comet
        columns = {'start_scan': 'scan',
                    'peptide': 'sequence',
                    'num_matched_peptides': 'num_candidates',
                    'assumed_charge': 'charge'}
        return columns

    def update_headers_pepxml(self, headers):
        return headers + ["modifications"]

    def add_modification_data_pepxml(self, spectrum_query, ns):
        modification_list = spectrum_query.findall('.//pepXML:modification_info', namespaces=ns)
        
        if modification_list:
            modification_info = [modification_list[0].attrib['modified_peptide']]
        else:
            modification_info = [None]
        return modification_info


class SpectraSTParser(ProteomicsDataParser):
    def __init__(self):
        super().__init__()
        self.engine = "SpectraST"

    def rename_columns(self):
        # Define column renaming logic specific to SpectraST
        columns = {'start_scan': 'scan',
                    'peptide': 'sequence',
                    'hits_num': 'num_candidates',
                    'p_value': 'p-value',
                    }
        return columns
    
    def update_headers_pepxml(self, headers):
        return headers + ["modifications"]

    
    def add_modification_data_pepxml(self, spectrum_query, ns):
        modification_list = spectrum_query.findall('.//pepXML:modification_info', namespaces=ns)
        
        if modification_list:
            modification_info = [modification_list[0].attrib['modified_peptide']]
        else:
            modification_info = [None]
        return modification_info


class TideParser(ProteomicsDataParser):
    def __init__(self):
        super().__init__()
        self.engine = "Tide"

    def rename_columns(self):
        # Define column renaming logic specific to Tide
        columns = {'exact p-value': 'p-value',
                   'exact_pvalue': 'p-value',
                    'distinct matches/spectrum': 'num_candidates',
                    'xcorr rank': 'hit_rank'}
        
        return columns


    def update_headers_pepxml(self, headers):
        return headers

    def add_modification_data_pepxml(self, spectrum_query, ns):
        return []


class MSFraggerParser(ProteomicsDataParser):
    def __init__(self):
        super().__init__()
        self.engine = "MSFragger"

    def rename_columns(self):
        # Define column renaming logic specific to MSFragger
        columns = {'start_scan': 'scan',
                    'peptide': 'sequence',
                    'hits_num': 'num_candidates',
                    'p_value': 'p-value',
                    }
        return columns
    


class MSGFParser(ProteomicsDataParser):
    def __init__(self):
        super().__init__()
        self.engine = "MSGF+"

    def rename_columns(self):
        # Define column renaming logic specific to MSGF+
        columns = {'start_scan': 'scan',
                    'peptide': 'sequence',
                    'hits_num': 'num_candidates',
                    'p_value': 'p-value',
                    }
        return columns



class SpTXTParser:
    """Process the data from sp.txt file produced by SpectraST"""

    PeptideModel = namedtuple(
                "Psm",
                 ["peptide","mz", "ints", "mz_freq", "int_sd"],
                 defaults=["", [], [], [], []])

    def __init__(self):
        self.lines = []
        

    def read_sptxt(self, filename):
        """load all lines from the file"""
        with open(filename, "r") as file:
            lines = file.readlines()
            self.lines = [line.rstrip() for line in lines]


    def read_lines(self):
        """Read full contents of the sptxt file"""
        peptide_models = {}
        k = 0

        for line in self.lines:

            if line[:4] == "Name":
                k += 1

                if k != 1:
                    if k % 10000 == 0:
                        print(f"{k}...", end="", flush=True)

                    peptide_models[current_model.peptide] = current_model
                    current_model = self.PeptideModel
                else:
                    current_model = self.PeptideModel

                current_model.peptide = line[6:]

            if len(line) > 1:
                if line[1].isnumeric():
                    current_model = self.add_pep_attrs(current_model, line)

        return peptide_models



    def read_only_peptide(self):
        """Read only peptide info from the lines"""
        peptides = []

        for line in self.lines:

            if line[:4] == "Name":
                #peptides.append(self.clean_sequence(line[6:]))
                peptides.append(str.rstrip(line[6:]))

        return np.array(peptides)



    @staticmethod
    def clean_sequence(seq):
        """remove unnecessary characters from the peptide sequence"""
        s1 = re.sub(r'{}.*?{}'.format(re.escape("["),re.escape("]")),'', seq)
        s1 = re.sub("-", "", s1)
        if s1[0] == 'n':
            s1 = s1[1:]
        return s1[:-2]



    @staticmethod
    def add_pep_attrs(model, line):
        """Add peptide info to the PeptideModel tuple"""
        mz, ints, annot, cv = line.split('\t')
        model.mz.append(eval(mz))
        model.ints.append(float(ints))
        model.mz_freq.append(eval(cv.split(" ")[0]))
        model.int_sd.append(float(cv.split(" ")[1].split("|")[1])*float(ints))

        return model
