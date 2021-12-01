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

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) # multiply sqrt(d_model) to embeded result


