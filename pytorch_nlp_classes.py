import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Tokenizer():
    """
    General purpose, simple, and naive tokenizer. Takes a string as splits it
    into words based on whitespace.
    param char_level: If True, every char is a token.
    param vocab_size: Keeps this many words, ordered by frequency.
    param oov_token: String used to represent out of vocabulary words.
    param filters: a string in which every char is to be filtered out of the
    text.
    """
    def __init__(self,
                 char_level=False,
                 vocab_size=None,
                 oov_token=None,
                 filters="""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""):
        self.char_level = char_level
        self.num_words = vocab_size
        self.oov_token = oov_token
        self.filters = filters
        self.word_counts = {}
        self.word_index = {}
        self.index_to_word = {}

    def fit(self,texts):
        """
        Fits the tokenizer. Allows it to build its internal dictionary.
        """
        for text in texts:
            if self.char_level:
                seq = text.lower()
            else:
                seq = self.text_to_words(text.lower())
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1

        self.word_index = {k[0]:i + 1 for i,k in enumerate(sorted(self.word_counts.items(),
                                                                 key= lambda x: x[1],
                                                                 reverse=True))}
        self.index_to_word = {v:k for k,v in self.word_index.items()}
        if self.oov_token is not None:
            self.word_index[self.oov_token] = len(self.word_index)+1
            self.index_to_word[len(self.word_index)+1] = self.oov_token

    def transform(self,texts):
        """
        Using the dictionaries built above, this transforms a sentence in a
        sequence of tokens.
        """
        tokenized = []
        for text in texts:
            if self.char_level:
                seq = text
            else:
                seq = self.text_to_words(text.lower())
            token_seq = []
            for w in seq:
                if w not in self.word_index:
                    if self.oov_token is not None:
                        token_seq.append(self.word_index[self.oov_token])
                else:
                    if self.num_words is not None:
                        if self.word_index[w] > self.num_words:
                            if self.oov_token is not None:
                                token_seq.append(self.word_index[self.oov_token])
                        else:
                            token_seq.append(self.word_index[w])
                    else:
                        token_seq.append(self.word_index[w])
            tokenized.append(token_seq)
        return tokenized

    def text_to_words(self,text):
        """
        Helper function. You shouldn't be calling this, but this breaks up a
        text by whitespace.
        """
        new_text = text
        for item in self.filters:
            new_text = new_text.replace(str(item),'')
        return new_text.split()

class Word2Vec(nn.Module):
    """
    Word2Vec class. Includes training via the skipgram method.
    param vocab_size: number of words in the vocabulary.
    param embedding_dim: dimension of the embedding vector.
    param device: device the model is built on, defaults to CPU.
    """
    def __init__(self,vocab_size,embedding_dim,device=torch.device('cpu')):
        super(Word2Vec,self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.to(device)
        self.output = nn.Linear(1, 1)
        self.output.to(device)

    def forward(self,words,contexts):
        """
        Forward pass through the model.
        param words: target words in the skipgram model.
        param contexts: candidate context words in the skipgram model.
        """
        words_tensor = self.embedding(words)
        contexts_tensor = self.embedding(contexts)
        similarity = F.cosine_similarity(words_tensor, contexts_tensor,dim=1)
        out = F.sigmoid(self.output(similarity.view(-1,1,1)))

        return out

    def represent(self,word):
        """
        Gives the vector representation of a word.
        param word: word to be represented.
        """
        return self.embedding(word)

    def similarity(self,w1,w2):
        """
        Calculates cosine similarity between two words, according to the model.
        The larger this is, more 'alike' are the words.
        param w1: first word of the cosine similarity
        param w2: second word of the cosine similarity
        """
        words_tensor = self.embedding(w1)
        contexts_tensor = self.embedding(w2)
        similarity = F.cosine_similarity(words_tensor.view(-1), contexts_tensor.view(-1),dim=0)
        return similarity

    def fit(self,
            words,
            contexts,
            targets,
            batch_size,
            epochs,
            optimizer,
            loss_function=nn.BCELoss(size_average=False)):
        """
        fits the Word2Vec model, according to the skipgram method.
        param words: target words.
        param contexts: candidate context words.
        param targets: binary classification for contexts.
        param batch_size: batch size to train on.
        param epochs: how many epochs to train.
        param optimizer: torch optimizer to use during training.
        param loss_function: loss function to use. Defaults to binary
        crossentropy.
        """
        history = []
        for _ in tqdm(range(epochs)):
            total_loss = torch.Tensor([0])
            for i in range(len(words[::batch_size])):
                batch_words = torch.tensor(words[i*batch_size:(i+1)*batch_size],
                                           dtype=torch.long,
                                           device=self.device)
                batch_contexts = torch.tensor(contexts[i*batch_size:(i+1)*batch_size],
                                              dtype=torch.long,
                                              device=self.device)
                # Explicitly zero all gradients
                self.zero_grad()
                preds = self(batch_words,batch_contexts).view(-1)
                loss = loss_function(preds,
                                     torch.tensor(targets[i*batch_size:(i+1)*batch_size],
                                                  dtype=torch.float,
                                                  device=self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(total_loss.item()/len(words))
            history.append(total_loss.item()/len(words))
        return history

    def test(self,
             words,
             contexts,
             targets,
             loss_function=nn.BCELoss(size_average=False)):
        """
        Helper method for OOF model evaluation.
        param words: target words.
        param contexts: candidate context words.
        param target: binary classification targets for candidate contexts.
        param loss_function: loss function to evaluate against.
        """

        return loss_function(preds,
                          torch.tensor(targets[i*batch_size:(i+1)*batch_size],
                                       dtype=torch.float,
                                       device=self.device)).item()

    def save_embedding(self,path):
        """
        Helper method for easy embedding layer serialization.
        param path: file in which state_dict gets saved.
        """
        torch.save(self.embedding.state_dict(),path)
        print("Saved the embedding layer's parameters in {}".format(path))

class Seq2Seq(nn.Module):
    """
    Sequence to sequence model class in pytorch. Currently implements a
    bidirectional, single layer, encoder and no attention mechanism.
    TODO: implement multi layer encoders, attention mechanism.
    param encoder_units: Size of encoding layer.
    param vocab_size: How many words are there in the vocabulary.
    param embedding_dim: Dimension of embedding layer.
    param input_length: max length of input/output sequences.
    param embedding_path: path to a serialized embedding layer, to ease transfer
    learning. if None, the embedding layer will be trained from scratch.
    param train_embedding: whether or not to include the embedding layer in
    gradient updates. useful if using pre-trained embedding layers.
    param device: torch device the model inhabits.
    """
    def __init__(self,
                 encoder_units,
                 vocab_size,
                 embedding_dim,
                 input_length,
                 embedding_path=None,
                 train_embedding=True,
                 device=torch.device('cpu')):
        super(Seq2Seq,self).__init__()
        self.device = device
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.encoder = nn.LSTM(embedding_dim,
                               encoder_units,
                               bidirectional=True,
                               batch_first=True)
        self.encoder.to(device)
        self.decoder = nn.LSTM(embedding_dim,2*encoder_units,batch_first=True)
        self.decoder.to(device)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim)
        if not train_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False
        # self.embedding.weight.requires_grad = train_embedding
        self.embedding.to(device)
        if embedding_path is not None:
            self.embedding.load_state_dict(torch.load(path))
        self.dense_layer = nn.Linear(2*encoder_units,vocab_size)
        self.dense_layer.to(device)

    def forward(self,encoder_inputs, decoder_inputs):
        """
        Forward pass for training. Encodes a sequence and then decodes its
        related target. Uses teacher forcing.
        param encoder_inputs: input sequences.
        param decoder_inputs: sequence to teacher force.
        """
        encoder_embedded = self.embedding(encoder_inputs)
        decoder_embedded = self.embedding(decoder_inputs)
        encoded, encoded_states = self.encoder(encoder_embedded)

        decoded, decoded_states = self.decoder(decoder_embedded,
                                               (encoded_states[0].view(1,-1,encoded.shape[2]),
                                               encoded_states[1].view(1,-1,encoded.shape[2])))
        log_probs = F.log_softmax(self.dense_layer(decoded),dim=2)
        return log_probs

    def fit(self,
            encoder_inputs,
            decoder_inputs,
            ground_truth,
            batch_size,
            epochs,
            optimizer,
            loss_function=F.nll_loss,
            size_average=False):
        """
        Training method.
        param encoder_inputs: input sequences.
        param decoder_inputs: sequences to teacher force.
        param ground_truth: target sequences. Usually the decoder_inputs shifted
        by one timestep.
        param batch_size: batch size.
        param epochs: number of training epochs.
        param optimizer: torch optimizer to use for the training process.
        param loss_function: loss function to be minimized. defaults to negative
        log likelihood.
        param size_average: whether or not to average each batch in
        loss_function. Only affects the printouts.
        """
        history=[]
        for _ in tqdm(range(epochs)):
            total_loss = torch.tensor([0],dtype=torch.float)
            for i in tqdm(range(len(encoder_inputs[::batch_size]))):
                loss = 0.
                batch_encode = torch.tensor(encoder_inputs[i*batch_size:(i+1)*batch_size],
                                            dtype=torch.long,
                                            device=self.device)
                batch_decode = torch.tensor(decoder_inputs[i*batch_size:(i+1)*batch_size],
                                            dtype=torch.long,
                                            device=self.device)
                self.zero_grad()
                preds = self.forward(batch_encode,batch_decode)
                for k in range(preds.shape[1]):
                    loss += loss_function(preds[:,k,:],torch.tensor(ground_truth[i*batch_size:(i+1)*batch_size],
                                                                    dtype=torch.long,
                                                                    device=self.device)[:,k],
                                          size_average=size_average)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(total_loss.item()/len(encoder_inputs))
            history.append(total_loss.item()/len(encoder_inputs))
        return history

    def encode(self,sequence):
        """
        Method to encode an input sequence and pass the states to a decoder.
        param sequence: input sequence.
        """
        encoder_embedded = self.embedding(sequence)
        encoded, encoded_states = self.encoder(encoder_embedded)
        return encoded_states

    def decode(self,encoded_state):
        """
        Method to decode an input state into a response sequence.
        param encoded_state: initial decoder state, contains information about
        input sequence.
        """
        state = encoded_state
        embedded = self.embedding(torch.tensor([0],device=self.device))
        decoded_seq = []
        decoded_token = torch.tensor([self.vocab_size+1])
        while (decoded_token != 0) and (len(decoded_seq) < self.input_length):
            decoded, state = self.decoder(embedded.view(1,1,-1),
                                          (state[0].view(1,1,-1),
                                           state[1].view(1,1,-1)))
            log_probs = F.log_softmax(self.dense_layer(decoded),dim=2).view(-1)
            decoded_token = log_probs.max(0)[1]
            decoded_seq.append(decoded_token.item())
            embedded = self.embedding(decoded_token)
        return decoded_seq

    def chat(self,tokenizer):
        """
        Human real-time testing method.
        param tokenizer: a tokenizer to turn text into sequences.
        """
        print("You can input sequences here and have them be encoded-decoded by the model.")
        while True:
            sentence = input()
            response = self.decode(self.encode(torch.tensor(tokenizer.transform([sentence]),dtype=torch.long,device=self.device)))
            words = []
            for item in response:
                words.append(tokenizer.index_to_word[item])
            print(' '.join(words))


    def load_embedding(self,path,word_index):
        """
        Loads pre-trained embeddings of the format 'word <components>', like the
        pretrained Stanford GloVe models.
        param path: path to the file.
        param word_index: dictionary mapping words to indexes.
        """
        embeddings_index = {}
        with open(path,'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        print("Loaded embedding located in {}".format(path))
