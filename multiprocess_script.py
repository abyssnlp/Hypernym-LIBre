'''
@author:srawat
'''

import re
import nltk
from nltk.tag import PerceptronTagger
import itertools
from tqdm import tqdm
chunk_patterns=r'''NP:{<DT>?<JJ.*>*<NN.*>+}
                      {<NN.*>+}
                '''

nounphrase_chunker=nltk.RegexpParser(chunk_patterns)

hearst_patterns=[ (r"(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)", "first"),
                (r"(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                (r"((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)", "last"),
                (r"(NP_\w+ (, )?including (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                (r"(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)", "first"),
                (r"(NP_\w+ (, )?which is (a|an) (example|class|kind) of NP_\w+ )","last"),
                (r"(NP_\w+ (, )?which is called [\w|\s]+ NP_\w+)","last"),
                (r"(NP_\w+ (, )?(is)?\s?a special case of NP_\w+)","last"),
                (r"(NP_\w+ (, )?is (a|an)? NP_\w+)","last"),
                (r"(NP_\w+ (, )?is a (member|part|given) of ?(the|a)? NP_\w+)","last"),
                (r"((features|properties) of? NP_\w+ (, )?such as (NP_\w+ ?(, )?(and |or )?)+)","first"),
                (r"(((U|u)nlike|(l|L)ike) (most|all|any|other) NP_\w+, NP_\w+)","first"),
                (r"(NP_\w+ including (NP_\w+ ?(, )?(and |or )?)+)","first"),
            ]

pos_tagger=PerceptronTagger()

def prepare(raw_text):
    raw_text=re.sub(r'\\','',raw_text)
    sentences=nltk.sent_tokenize(raw_text.strip())
    sentences=[nltk.word_tokenize(sent) for sent in sentences]
    sentences=[pos_tagger.tag(sent) for sent in sentences]
    return sentences

def chunk(raw_text):
    sentences=prepare(raw_text.strip())
    all_chunks=[]
    for sentence in sentences:
        chunks=nounphrase_chunker.parse(sentence)
        all_chunks.append(prepare_chunks(chunks))
    all_sentences=[]
    for raw_sentence in all_chunks:
        sentence = re.sub(r"(NP_\w+ NP_\w+)+",lambda m: m.expand(r'\1').replace(" NP_", "_"),raw_sentence)
        all_sentences.append(sentence)

    return all_sentences

def prepare_chunks(chunks):
    terms=[]
    for chunk in chunks:
        label=None
        try:
            label=chunk.label()
        except:
            pass
        if label is None:
            token=chunk[0]
            terms.append(token)
        else:
            np='NP_'+'_'.join([a[0] for a in chunk])
            terms.append(np)
    return ' '.join(terms)


def remove_np_term(term):
    return term.replace('NP_','').replace('_',' ')

def find_hyponyms(rawtext):
    hypo_hypernyms=[]
    np_tagged_sentences=chunk(rawtext)

    for sentence in np_tagged_sentences:
        for (hearst_pattern,parser) in hearst_patterns:
            matches=re.search(hearst_pattern,sentence)
            if matches:
                # matched_pattern=hearst_pattern
                match_str=matches.group(0)
                nps=[a for a in match_str.split() if a.startswith('NP_')]

                if parser=='first':
                    hypernym=nps[0]
                    hyponyms=nps[1:]
                else:
                    hypernym=nps[-1]
                    hyponyms=nps[:-1]
                for i in range(len(hyponyms)):
                    hypo_hypernyms.append((remove_np_term(hyponyms[i]),remove_np_term(hypernym)))
    return hypo_hypernyms




def extractHearstWikipedia(input):
    extractions=[]
    lines_processed=0

    with open('data/'+input,'r') as f:
        for  line in tqdm(f):
            lines_processed+=1
            line=line.strip()
            if not line:
                continue
            line_split=line.split('\t')
            # Uncomment to run on full dataset
            # Changed to work on subset of the data
            # sentence,lemma_sent=line_split[0].strip(),line_split[1].strip()
            sentence=line_split[0]
            hypo_hyper_pairs=find_hyponyms(sentence)
            extractions.append(hypo_hyper_pairs)

            if lines_processed%1000==0:
                print('Lines Processed: {}'.format(lines_processed))
    extractions=[x for x in extractions if x!=[]]
    extractions=list(itertools.chain.from_iterable(extractions))
    return extractions

import multiprocessing
import os

# Make Directory to store file chunks
os.mkdir('data/')

# Split Corpus into smaller Files
lines_per_file = 100000
smallfile = None
with open('combined_corpus.txt') as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = 'data_{}.txt'.format(lineno + lines_per_file)
            smallfile = open('data/'+small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()

files=[data for data in os.listdir('data/') if data.startswith('data_')]

cpu_count=multiprocessing.cpu_count()
pool=multiprocessing.Pool(processes=cpu_count-1)
par_extractions=pool.map(extractHearstWikipedia,files)
par_extractions=list(itertools.chain.from_iterable(par_extractions))
#def writer(dest_filename,queue,stop_token):
with open('main_hypernyms.txt','w') as f:
        for (hypo,hyper) in par_extractions:
            try:
                f.write(hypo+'\t'+hyper+'\n')
            except ValueError:
                pass