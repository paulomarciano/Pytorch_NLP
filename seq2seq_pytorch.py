import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_nlp_classes as nlp
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
    with open('./Cornell/train.enc','r') as f:
        train_enc_list = []
        for line in f:
            train_enc_list.append(line.lower())
    with open('./Cornell/train.dec','r') as f:
        train_dec_list=[]
        for line in f:
            train_dec_list.append(line.lower())

    tokenizer = nlp.Tokenizer(vocab_size=10000, oov_token='<UNK>')
    tokenizer.fit(np.hstack([train_enc_list,train_dec_list]))

    s2s=nlp.Seq2Seq(16,
                    len(tokenizer.word_index)+1,
                    100,
                    40,
                    train_embedding=False,
                    device=torch.device('cuda:0'))
    s2s.load_embedding('./Cornell/glove.6B.100d.txt',tokenizer.word_index)

    # Preparing inputs and targets:
    enc_seq = tokenizer.transform(train_enc_list)
    dec_seq = tokenizer.transform(train_dec_list)
    for i in range(len(enc_seq)):
        enc_seq[i] = pad_seq(enc_seq[i],0,40)
        dec_seq[i] = pad_seq(dec_seq[i],0,40)
    targets=dec_seq.copy()
    for i in range(len(dec_seq)):
        dec_seq[i].pop()
        dec_seq[i].insert(0,0)

    s2s.fit(enc_seq,
            dec_seq,
            targets,
            64,
            100,
            optimizer=optim.RMSprop(filter(lambda p: p.requires_grad,
                                           s2s.parameters()),
                                    lr=1e-3))

    torch.save(s2s.state_dict(), s2s.pt)

    s2s.chat(tokenizer)
