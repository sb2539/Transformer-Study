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


d_model = 512
text = "I am a student"
spacy_en = spacy.load('en_core_web_sm') # using full name 'en' -> 'en_core_web_sm'
dropout = 0.1

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

vocab = tokenize(text)
print(vocab, len(vocab))


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
        div_term = torch.exp(torch.arange(0, d_model, 2)*
                             -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad = False)
        return self.dropout(x)

# show the graph
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" %p for p in [4,5,6,7]])
