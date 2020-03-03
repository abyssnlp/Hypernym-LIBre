import numpy as np
import os
import scipy.sparse as sp

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
matrix=matrix.todok()

sum_rows=matrix.sum(axis=1).A[:, 0]  
sum_cols=matrix.sum(axis=0).A[0,:]

N=len(sum_rows)
def predict(hypo,hyper):
    L=vocab.get(hypo,0)
    R=vocab.get(hyper,0)
    return matrix[(L,R)]

# Check Raw Count Model
#prob('PCI','way')
# Raw count Probability
total_extractions=matrix.sum()

def prob(hypo,hyper):
    return predict(hypo,hyper)/total_extractions

# Raw count matrix
raw_count_matrix=np.zeros(matrix.shape)
for (l,r) in matrix.keys():
    raw_count=matrix[(l,r)]/total_extractions
    raw_count_matrix[(l,r)]=raw_count

# PPMI matrix
def pmi(hypo,hyper):
    return np.log(N)+np.log(matrix[(hypo,hyper)])-np.log(sum_rows[hypo])-np.log(sum_cols[hyper])
pmi_matrix=np.zeros(matrix.shape)
for (l,r) in matrix.keys():
    pmi_value=pmi(l,r)
    pmi_matrix[(l,r)]=pmi_value
ppmi_matrix=np.clip(pmi_matrix,0,1e12)


# Svd over Raw count matrix
k=10 # from FB AI Paper
# TODO:This can also be optimized using Hyperparameter tuning
# k=[5,10,15,20,25,50,100,150,200,250,300,500,1000]
from scipy.sparse import linalg
U,S,V=linalg.svds(raw_count_matrix.tocsr(),k=k)
U=U.dot(np.diag(S))
V=V.T

# SVD over PPMI Matrix
U_p, S_p, V_p = linalg.svds(ppmi_matrix.tocsr(), k)
