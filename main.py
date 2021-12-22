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
        self.proj = nn.Linear(d_model, vocab)  # 선형 변환

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim = -1)  # dim = -1이 의미하는 것이 무엇인가

# N 만큼의 identical layers 생성
def clones(module, N) :
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) # 모듈 리스트 리턴

# Encoder class
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # 모듈리스트에서 layer 가져옴
        self.norm = LayerNorm(layer.size) # 층 정규화

    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# LayerNormalization class
class LayerNorm(nn.Module):
# Construct a layernorm module (See citation for details)
    def __init__(self, features, eps = 1e-6):  # eps = 분모 0 방지하는 매우 작은 값 (지수표기법 : 0.000001
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) # 초기 값 1인 학습가능한 감마 파라미터
        self.b_2 = nn.Parameter(torch.zeros(features)) # 초기 값 0인 학습가능한 베타 파라미터
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 평균 산출
        std = x.std(-1, keepdim=True) # 분산 산출
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# SublayerConnection class

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))  # skip connection 부분
## FeedForward class

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 가중치 행렬 w1
        self.w_2 = nn.Linear(d_ff, d_model) # 가중치 행렬 w2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # FFN(X) = RELU(XW_1 + b_1)W_2 + b_2

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
