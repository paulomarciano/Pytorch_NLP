import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_nlp_classes as nlp
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm

def pad_seq(seq,symbol,maxlen):
    if len(seq)>maxlen:
        rseq = seq[:maxlen-1]
        rseq.append(symbol)
        return rseq
    while len(seq) < maxlen:
        seq.append(symbol)
    return seq

if __name__ == "__main__":
    df = pd.DataFrame()
    for data_file in tqdm(glob('./OpenSubtitles/*.csv')):
        df = df.append(pd.read_csv(data_file))

    df = df.reset_index(drop=True)

    tokenizer_input = nlp.Tokenizer(oov_token='<UNK>')
    tokenizer_input.fit(df.Portuguese.values)

    tokenizer_output = nlp.Tokenizer(oov_token='<UNK>')
    tokenizer_output.fit(df.English.values)

    s2s = nlp.Seq2Seq(16,
                      len(tokenizer_input.word_index)+1,
                      len(tokenizer_output.word_index)+1,
                      100,
                      40,
                      embedding_path_in=None,
                      embedding_path_out=None,
                      train_embedding_in=False,
                      train_embedding_out=False,
                      device=torch.device('cuda:0'))

    s2s.load_embedding('./Cornell/glove.6B.100d.txt',tokenizer_output.word_index,which='decoder')
    s2s.load_embedding('./Cornell/glove_s100.txt',tokenizer_intput.word_index,which='encoder')
