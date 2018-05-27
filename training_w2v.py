import pytorch_nlp_classes as nlp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
import json
import datetime
from tqdm import tqdm
from keras.preprocessing.sequence import skipgrams, make_sampling_table

embedding_dim = 100
batch_size = 10000
epochs = 5

print("Loading Files... {}".format(datetime.datetime.time(datetime.datetime.now())))
fsp = pd.read_csv('./data/folha_de_sao_paulo/articles.csv')
docs = list(fsp.text.dropna())
for folder in tqdm(os.listdir('./data/Wiki Text')):
    gl = glob.glob(os.path.join('./data/Wiki Text', folder) + '/*')
    for filename in gl:
        with open(filename, 'r') as f:
            for line in f:
                docs.append(json.loads(line)['text'])

print("Tokenizing docs... {}".format(datetime.datetime.time(datetime.datetime.now())))
tokenizer = nlp.Tokenizer(oov_token='<UNK>')
tokenizer.fit(docs)

docs = tokenizer.transform(docs)
vocab_size = len(tokenizer.word_index)+1

print("Creating the model... {}".format(datetime.datetime.time(datetime.datetime.now())))
w2v = nlp.Word2Vec(vocab_size,embedding_dim,device=torch.device('cuda:0'))
sampling_table = make_sampling_table(vocab_size)

couples = []
targets = []

print("Training... {}".format(datetime.datetime.time(datetime.datetime.now())))
for t in range(epochs):
    print("epoch #: {}".format(t))
    for i,doc in enumerate(docs):
        new_couples, new_targets = skipgrams(doc,
                                             vocab_size,
                                             sampling_table=sampling_table)
        couples.extend(new_couples)
        targets.extend(new_targets)

        if (i > 0 and i % 20000 == 0) or (i >= len(docs)-1):
            words, contexts = zip(*couples)

            history = w2v.fit(words,
                              contexts,
                              targets,
                              batch_size,
                              epochs=1,
                              optimizer=optim.Adam(w2v.parameters(),lr=1e-3))

            couples = []
            targets = []

w2v.save_embedding('./teste.pt')
