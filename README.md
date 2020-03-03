# Hypernym-LIBre
A free Web-based Corpus for Hypernym Detection

This repository holds the code and other relevant details for the paper submitted to WAC which is held along with LREC 2020. 

### **Authors** : Rawat, S., Rico, M., Corcho O.

Description
-----------
Detecting hierarchial relationships between terms in text is a key area of research in the field of knowledge representation and NLP. In our paper, describe a new web-based corpus for extracting hypernyms and compare it with a corpus used for the current state-of-the-art but is not freely available.  
The corpus is a combination of the [UMBC Corpus](https://ebiquity.umbc.edu/blogger/2013/05/01/umbc-webbase-corpus-of-3b-english-words/) and [Wikipedia](https://dumps.wikimedia.org/enwiki/). We apply our own pre-processing and post-processing methodologies for extracting hypernym-hyponym pairs from the text using [Hearst Patterns](https://dl.acm.org/doi/10.3115/992133.992154) (Marti Hearst, 1992). 

Location
---------
The corpus is provided in two formats. One is the raw text format and the other is its part-of-speech tagged and dependency parsed version. 

Raw Text Format of Hypernym-LIBre:   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3662204.svg)](https://doi.org/10.5281/zenodo.3662204)

Part-of-Speech tagged and Dependency annotated version of Hypernym-LIBre:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3689303.svg)](https://doi.org/10.5281/zenodo.3689303)

Hypernym pairs extracted from Hypernym-LIBre using Hearst patterns:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3695237.svg)](https://doi.org/10.5281/zenodo.3695237)



### Resource Description
1. Raw Text format of Hypernym LIBre:  
   Number of Files: 288 files of ~110MB each
   Size: 11.3GB compressed, 32GB uncompressed
2. PoS-tagged and Dependency annotated format of Hypernym-LIBre:
   Number of Files: 442 files of ~180MB each
   Size: 15GB compressed, 80.6GB uncompressed

### File Description
1. *multiprocess_script.py* : Extract Hearst patterns from multiple chunks of Hypernym-LIBre using multiprocessing
2. *hearst_counts_alternate.py*: Create a dictionary of extractions using Hearst Patterns and get the frequencies
3. *raw_count_model.py*: Create a compressed sparse row matrix from the above file and calculate the raw probability model 
4. *PPMI_model.py*: Create a positive pointwise mutual information matrix from the raw count matrix
5. *SVD_raw_count_model_ppmi.py*: Apply matrix factorization using SVD on both the raw count matrix and the PPMI matrix to get low rank embeddings. This helps in creating similar representations for similar words.
   
Wikipedia extractor used to convert wikidump into text file is [here](http://wiki.apertium.org/wiki/Wikipedia_Extractor).