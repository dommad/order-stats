"""Module for processing data from spectral libraries in format .sptxt"""
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
import pandas as pd
from .utils import _is_numeric, ParserError

FILE_FORMATS = ['pep.xml', 'txt', 'mzid']
SEARCH_ENGINES = ['Comet', 'SpectraST', 'Tide', 'MSFRagger', 'MSGF+']


class PSMParser(ABC):
    def __init__(self, decoy_tag='decoy'):
        self.decoy_tag = decoy_tag


    @abstractmethod
    def rename_columns(self):
        pass


    def parse(self, file_name):
        after_dots = file_name.lower().split('/')[-1].split('.')
        
        if after_dots[-2:] == ['pep', 'xml']:
            file_ext = 'pepxml'
        elif after_dots[-1] in FILE_FORMATS:
            file_ext = after_dots[-1]
        else:
            raise ValueError(f"Unsupported file extension for the file: {file_name}")

        return getattr(self, f"parse_{file_ext}")(file_name)


    def parse_pepxml(self, file_name):
        """Parses pepxml (Comet) and outputs pandas dataframe"""
        try:

            # Define the namespace used in the XML
            ns = {'pepXML': 'http://regis-web.systemsbiology.net/pepXML'}

            def get_spectrum_queries():
                for event, elem in ET.iterparse(file_name, events=('start', 'end')):
                    if event == 'start' and elem.tag.endswith('spectrum_query'):
                        spectrum_info = elem.attrib

                    elif event == 'end' and elem.tag.endswith('search_hit'):
                        search_hit_info = self.parse_spectrum_info(elem.iter())
                        analysis_result_list = elem.findall('.//pepXML:analysis_result', namespaces=ns)
                        analysis_result_info = {}

                        for analysis_result in analysis_result_list:
                            analysis_tag = analysis_result.attrib.get('analysis', '')
                            raw_analysis_info = self.parse_spectrum_info(analysis_result.iter())
                            cur_analysis = {f"{analysis_tag}_{key}": val for key, val in raw_analysis_info.items()}
                            analysis_result_info.update(cur_analysis)

                        combined = {**spectrum_info, **search_hit_info, **analysis_result_info}
                        yield combined
                        elem.clear()

            df = pd.DataFrame(get_spectrum_queries())
            df.rename(columns=self.rename_columns(), inplace=True)
            df['is_decoy'] = [True if self.decoy_tag in row.protein.lower() else False for row in df.itertuples()]
            df['modifications'] = df.apply(lambda row: (row['position'], row['mass']), axis=1)

            return df
        
        except ParserError as err:
            print(f"Error parsing file: {err}")
            return None


    def turn_strings_into_floats(self, x):
        return {k: float(v) if _is_numeric(v) else v for k, v in x.items()}
        

    def parse_spectrum_info(self, spectrum_info):
        master_dict = {}
        for item in spectrum_info:
            cur_dict = item.attrib
            if all(key in cur_dict for key in ['name', 'value']):
                master_dict.update({cur_dict['name']: cur_dict['value']})
                del cur_dict['name']
                del cur_dict['value']
            
            master_dict.update(cur_dict)

        return self.turn_strings_into_floats(master_dict)


    def parse_txt(self, file_name, sep='\t'):
        # Implement common parsing logic
        df = pd.read_csv(file_name, sep=sep)
        # renaming will be handled by individual engines
        df.rename(columns=self.rename_columns(), inplace=True)
        df['is_decoy'] = [True if self.decoy_tag in row.protein.lower() else False for row in df.itertuples()]
        return df


    def parse_mzid(self, xml_file_path):
      
        def get_spectra():
            peptides = {}
            peptide_evidence = {}
            db_proteins = {}
            # Iterate over XML elements as they are parsed
            for event, elem in ET.iterparse(xml_file_path, events=('start', 'end')):
                if event == 'start':
                    
                    if elem.tag.endswith("MzIdentML"):
                        ns = {'mzIdentML': elem.tag.split('}')[0][1:]}
                    
                    elif elem.tag.endswith("DBSequence"):
                        protein_info = elem.attrib
                        protein_info['protein_name'] = protein_info['accession']
                        prot_dict = {protein_info['id']: protein_info}
                        db_proteins.update(prot_dict)

                    # Process start events to collect Peptide information
                    elif elem.tag.endswith("Peptide") and elem.attrib.get("id"):
                        pep_info = self.parse_spectrum_info(elem.iter())
                        pep_dict = {pep_info['id']: pep_info}
                        peptides.update(pep_dict)
                    
                    elif elem.tag.endswith("PeptideEvidence") and elem.attrib.get("id"):
                        evidence_dict = self.parse_spectrum_info(elem.iter())
                        peptide_evidence.update({evidence_dict['id']: evidence_dict})
                        

                elif event == 'end':
                    # Process end events to collect SpectrumIdentificationResults
                    combined_info = {}
                    if elem.tag.endswith("SpectrumIdentificationResult") and elem.attrib.get("id"):
                        spectrum_attrib = elem.attrib

                        for spectrum_identification_item in elem.findall('.//mzIdentML:SpectrumIdentificationItem', ns):
                            psm_info = self.parse_spectrum_info(spectrum_identification_item.iter())
                            mod_info = peptides.get(psm_info['peptide_ref'], {})
                            # TODO: consider moving the logic of protein info to an abstract function, Comet handles it differently than MSGF+
                            peptide_evidence_info = peptide_evidence.get(psm_info.get('peptideEvidence_ref', ""), "")
                            protein_info = db_proteins.get(peptide_evidence_info.get('dBSequence_ref', ""), {})
                            combined_info = {**spectrum_attrib, **psm_info, **mod_info, **protein_info, **peptide_evidence_info}
                            yield combined_info

                        elem.clear()

        # Create a DataFrame
        df = pd.DataFrame(get_spectra())
        df.rename(columns=self.rename_columns(), inplace=True)
        df['is_decoy'] = [True if self.decoy_tag in row.protein else False for row in df.itertuples()]

        return df


class CometParser(PSMParser):
    def __init__(self):
        super().__init__()

    def rename_columns(self):
        # Define column renaming logic specific to Comet
        columns = { # pepxml columns
                    'start_scan': 'scan',
                    'peptide': 'sequence',
                    'num_matched_peptides': 'num_candidates',
                    'expect': 'e_value',
                    'modified_peptide': 'modifications',
                    #txt columns
                    'e-value': 'e_value',
                    'plain_peptide': 'sequence',
                    # mzid columns
                    'spectrumID': 'scan',
                    'rank': 'hit_rank',
                    'chargeState': 'charge',
                    'peptide_ref': 'sequence',
                    'Comet:expectation value': 'e_value',
                    'peptideEvidence_ref': 'protein',
                    'Comet:xcorr': 'xcorr',
                    }
        return columns


class SpectraSTParser(PSMParser):
    def __init__(self):
        super().__init__()

    def rename_columns(self):
        # Define column renaming logic specific to SpectraST
        columns = {'start_scan': 'scan',
                    'peptide': 'sequence',
                    'hits_num': 'num_candidates',
                    }
        return columns
    

class TideParser(PSMParser):
    def __init__(self):
        super().__init__()

    def rename_columns(self):
        # Define column renaming logic specific to Tide
        columns = {'exact p-value': 'p_value',
                    'distinct matches/spectrum': 'num_candidates',
                    'xcorr rank': 'hit_rank',
                    'protein id': 'protein',}
        
        return columns


class MSFraggerParser(PSMParser):
    def __init__(self):
        super().__init__()

    def rename_columns(self):
        # Define column renaming logic specific to MSFragger
        columns = {'SpectrumID': 'scan',
                'Rank': 'hit_rank', 
                'Peptide_Sequence': 'sequence',
                'Modifications': 'modifications'}

        return columns
    

class MSGFParser(PSMParser):
    def __init__(self):
        super().__init__()

    def rename_columns(self):
        # Define column renaming logic specific to MSGF+
        columns = {'start_scan': 'scan',
                    'peptide': 'sequence',
                    'hits_num': 'num_candidates',
                    'MS-GF:EValue': 'e_value',
                    'protein_name': 'protein',
                    }
        return columns


class ParamFileParser:

    def __init__(self, param_input) -> None:
        self.param_input = param_input
        self.param_dict: dict = {}
    
    def read_param_file(self):

        with open(self.param_input, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    
            for idx, line in enumerate(lines, 1):
                params = line.rstrip().split(' ')
                params = tuple(float(x) for x in params)
                self.param_dict[idx] = params