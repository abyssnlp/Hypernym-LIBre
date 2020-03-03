# Positive Pointwise Mutual Information on the Hearst Patterns extracted
# Derived from the raw_count model

import numpy as np
import os
import scipy.sparse as sp
from tqdm import tqdm

def load_matrix(filename):
    objects=['<OOV>']
    vocab={'<OOV>':0}
    i=[]
    j=[]
    v=[]
    with open(filename,'r') as f:
        for line in f:
            line=line.encode().decode('utf-8').strip()
            hypo,hyper,count=line.split('\t')
            if hypo=='hypos' and hyper=='hypers':
                continue
            if hypo not in vocab:
                vocab[hypo]=len(vocab)
                objects.append(hypo)
            if hyper not in vocab:
                vocab[hyper]=len(vocab)
                objects.append(hyper)
            i.append(vocab[hypo])
            j.append(vocab[hyper])
            v.append(int(count))
    f.close()    
    matrix=sp.csr_matrix((v,(i,j)),shape=(len(vocab),len(vocab)),dtype=np.float64)
    return matrix,objects,vocab

# Sparse Row matrix from Hearst Pattern Frequencies
matrix,objects,vocab=load_matrix('wiki_hearst_patterns_counts.tsv') 
sum_rows=matrix.sum(axis=1).A[:, 0]  
sum_cols=matrix.sum(axis=0).A[0,:]

# Matrix to DOK format
matrix2=matrix.todok()
sum_rows=matrix2.sum(axis=1).A[:, 0]  
sum_cols=matrix2.sum(axis=0).A[0,:]
N= sum_cols.sum()
# PMI Model
# Transform count matrix to PMI matrix

# PMI 
def pmi(hypo,hyper):
    return np.log(N)+np.log(matrix2[(hypo,hyper)]/(sum_rows[hypo]*sum_cols[hyper]))

def pmi2(hypo,hyper):
    return np.log(N)+np.log(matrix2[(hypo,hyper)])-np.log(sum_rows[hypo])-np.log(sum_cols[hyper])

# def pmi2(hyper_tuple):
#     hypo,hyper=hyper_tuple
#     return np.log2((matrix2[(hypo,hyper)])/(sum_rows[hypo])*(sum_cols[hyper]))

# pmi(vocab['PCI'],vocab['way'])    
import multiprocessing as mp

# PMI matrix
pmi_matrix=np.zeros(shape=(len(vocab),len(vocab)))
# cpu_count=mp.cpu_count()
# pool=mp.Pool(processes=cpu_count-1)
#pool.map(pmi2,list(matrix2.keys()))

# Fill in the Values
for (hypo,hyper) in matrix2.keys():
    pmi_value=pmi(hypo,hyper)
    pmi_matrix[(hypo,hyper)]=pmi_value

from numpy import count_nonzero
count_nonzero(np.clip(pmi_matrix,0,1e12))
