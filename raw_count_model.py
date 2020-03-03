
# Implementation of Raw count model of the Hearst Patterns

# The matrix for this will be the hypo-hyper matrix with freq as weights

# Vocabulary for both the hyponym and the hypernym remains the same
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

def predict(hypo,hyper):
    L=vocab.get(hypo,0)
    R=vocab.get(hyper,0)
    return matrix[(L,R)]

# Check Raw Count Model
predict('PCI','way')

# Raw count Probability
total_extractions=matrix.sum()

def prob(hypo,hyper):
    return predict(hypo,hyper)/total_extractions

# Raw count matrix
raw_count_matrix=np.zeros(matrix.shape)
for (l,r) in matrix.keys():
    raw_count=matrix[(l,r)]/total_extractions
    raw_count_matrix[(l,r)]=raw_count
