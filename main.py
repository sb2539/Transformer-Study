import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import spacy
seaborn.set_context(context="talk")
##%matplotlib inline

# tokenize
d_model = 512
text = "I am a student"
spacy_en = spacy.load('en_core_web_sm') # using full name 'en' -> 'en_core_web_sm'
dropout = 0.1

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

vocab = tokenize(text)
print(vocab, len(vocab))
 ###############################

# encoderdecoder class

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# class generator

class Generator(nn.Module):
    # Define standard linear + softmax generation step"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim = -1)  # dim = -1이 의미하는 것이 무엇인가

## Word embedding class

class Embeddings(nn.Module) :
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  # (seq_len, d_model) embedding
        self.d_model = d_model
## is it need??
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) # multiply sqrt(d_model) to embeded result

## positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)   # dropout 0.1

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)     # max_len * d_model 0으로 채워진 텐서
        position = torch.arange(0, max_len).unsqueeze(1)  # max_len * 1 크기의 텐서 생성
        div_term = torch.exp(torch.arange(0, d_model, 2)*  # 1/10000^(2i/d_mdel)
                             -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # pe가 학습되지 않도록 함

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad = False)
        return self.dropout(x)
    #def forward(self, x):
    #    x = x + self.pe[:, :x.size(1)]
    #    return self.dropout(x)

# show the graph
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" %p for p in [4,5,6,7]])
plt.show()
