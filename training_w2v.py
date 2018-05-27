import pytorch_nlp_classes as nlp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from keras.preprocessing.sequence import skipgrams, make_sampling_table

embedding_dim = 100
batch_size = 10000
epochs = 20

fsp = pd.read_csv('/home/paulo/projetos/Seq2Seq/data/folha_de_sao_paulo/articles.csv')
docs = list(fsp.text.dropna())

tokenizer = nlp.Tokenizer(oov_token='<UNK>')
tokenizer.fit(docs)
docs = tokenizer.transform(docs)
vocab_size = len(tokenizer.word_index)+1
w2v = nlp.Word2Vec(vocab_size,embedding_dim,device=torch.device('cuda:0'))
sampling_table = make_sampling_table(vocab_size)

couples = []
targets = []
for doc in docs[:10000]:
    new_couples, new_targets = skipgrams(doc,
                                         vocab_size,
                                         sampling_table=sampling_table)
    couples.extend(new_couples)
    targets.extend(new_targets)

words, contexts = zip(*couples)

history = w2v.fit(words,
                  contexts,
                  targets,
                  batch_size,
                  epochs=epochs,
                  optimizer=optim.Adam(w2v.parameters(),lr=1e-3))

w2v.save_embedding('./teste.pt')
