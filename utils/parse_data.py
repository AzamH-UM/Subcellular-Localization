#Functions for parsing data
import os
import sys
import pandas as pd
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser 
from Bio.PDB.PDBIO import PDBIO
pd.options.mode.chained_assignment = None  # default='warn'

#Read deeploc_df
def read_deep_loc(data_path = 'data/deeploc/deeploc.xlsx'):
    deeploc_df = pd.read_excel(data_path, header = 0, index_col = 0)
    return deeploc_df

#Convert mmcif to pdb
def mmcif_to_pdb(protein, cif_file, pdb_file):
    parser = MMCIFParser()
    structure = parser.get_structure(protein, cif_file)
    io=PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)
    return

#Add alphafold paths to deeploc_df
def get_deep_loc_af2(deeploc_df, data_dir = 'data/deeploc_af2'):
    deeploc_df['PDB Path'] = ''
    
    #Loop through all indices
    for i, protein in enumerate(deeploc_df.Protein):

        #Get pdb path of protein
        protein = protein.split('-')[0]
        cif_file = os.path.join(data_dir, f'AF-{protein}-F1-model_v4.cif')
        pdb_file = os.path.join(data_dir, f'AF-{protein}-F1-model_v4.pdb')

        #Check if mmCIF exists
        if os.path.isfile(cif_file):
    
            #Convert to PDB
            if not os.path.isfile(pdb_file):
                mmcif_to_pdb(protein, cif_file, pdb_file)

            #Add path to dataframe
            deeploc_df['PDB Path'].iloc[i] = pdb_file
    
    #Remove structures without PDB
    deeploc_df = deeploc_df.replace('', np.nan).dropna(how = 'any')
    return deeploc_df

#Split deeploc into train and test set
def split_deeploc(deeploc_af2_df):
    train_deeploc_af2_df = deeploc_af2_df[deeploc_af2_df['Split'] == 'train']
    test_deeploc_af2_df = deeploc_af2_df[deeploc_af2_df['Split'] == 'test']
    return train_deeploc_af2_df, test_deeploc_af2_df