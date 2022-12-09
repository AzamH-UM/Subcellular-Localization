import Bio.PDB
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float32)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

pdb_files = glob("./Data/deeploc_af2/*.pdb")

def save_outs(pdb_files):
    print(len(pdb_files))
    for i in tqdm(range(len(pdb_files))):
        pdb_file = pdb_files[i]
        pdb_code = pdb_file.split("-")[-3].split(".")[0]
        
        structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_file)
        model = structure[0]
        
        dist_matrix = calc_dist_matrix(model["A"], model["A"])
        contact_map = dist_matrix < 12.0
        dist_matrix = dist_matrix/np.max(dist_matrix)
        contact_map = contact_map.astype(np.float32)

        dist_matrix = cv2.resize(dist_matrix, (512, 512), interpolation = cv2.INTER_NEAREST)
        contact_map = cv2.resize(contact_map, (512, 512), interpolation = cv2.INTER_NEAREST)

        out_mat = np.stack((contact_map, dist_matrix))
        out_file = pdb_file.replace(".pdb", ".npy")
        np.save(out_file, out_mat)
    return 0

from multiprocessing import Pool, Process
def assign_work(workers):
    procs = []
    files = glob("./Data/deeploc_af2/*.pdb")
    # workers = workers
    chunk = len(files)//workers
    print(len(files), chunk)
    
    for i in range(workers): 
        if(i==workers-1):
            f = files[i*chunk:]
        else:
            f = files[i*chunk:(i+1)*chunk]
        
        proc = Process(target=save_outs, args=([f]))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

if __name__ == '__main__':
    assign_work(16)