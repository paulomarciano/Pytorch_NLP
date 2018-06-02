import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_nlp_classes as nlp
from tqdm import tqdm

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

    s2s.load_state_dict(torch.load('s2s.pt'))

    s2s.chat(tokenizer)
