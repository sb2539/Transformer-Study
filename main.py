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
""" 임시로 넣은 토큰 부분 """
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
# src_mask 가 패딩 마스크고, tgt_mask 가 룩어헤드 마스크라고 생각이 듬
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder # 인코더
        self.decoder = decoder # 디코더
        self.src_embed = src_embed  # 입력 문장 임베디드
        self.tgt_embed = tgt_embed  # 타켓 문장 임베디드
        self.generator = generator  # 제너레이터

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)  # 인코더 와 같은 메모리 가져야 해서 다음과 같이 했나봄

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask) # 인코드 함수는 임베디드 된 입력 문장에 대해
                                                         # 마스크 연산과 함께 인코딩

    def decode(self, memory, src_mask, tgt, tgt_mask): # 디코드 함수는 임베디드 된 타켓 문장에 대해
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask) # 패딩 마스크,
                                                                            # 룩어헤드 마스크 연산과
                                                                # 함께 디코딩

# class generator
"""디코더에서 각 timestep 별로 softmax 하여 확률값 반환"""
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
        self.layers = clones(layer, N) # 모듈리스트에서 layer N개만큼 가져옴
        self.norm = LayerNorm(layer.size) # 층 정규화

    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask) # 이렇게 짜면 레이어가 층으로 쌓이는건가?
        return self.norm(x) # 층 정규화 연산결과를 내보냄

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
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # 피드포워드 계산 결과 내보냄

# SublayerConnection class

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size) # 층 정규화 결과
        self.dropout = nn.Dropout(dropout) # 드롭아웃

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))  # skip connection 해서
                                                        # 층 정규화 결과와 입력 행렬 더해주고 리턴

# EncoderLayer class

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and FFN"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # 클론 함수 통해서 서브레이어 복사
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # 첫번째 층 셀프 어탠션
        return self.sublayer[1](x, self.feed_forward)  # 두번 째 층 피드포워드 출력만 다음 add&norm 연산에 필요하니깐

## decoder class
class Decoder(nn.Module):
    "Generic N layer decoder  with masking"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  # 층 정규화

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask) # 인코더 클래스와 동일 but 미리보기 마스크 추가
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (define below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask)) # 디코더의 첫번째 레이어 셀프어탠션 함
        x = self.sublayer[1](x, lambda x : self.src_attn(x, m, m, src_mask))  # 디코더의 두번째 레이어 셀프어탠션 아님 key, value는 인코더 출력
        return self.sublayer[2](x, self.feed_forward) # 피드 포워드 까지 한 결과를 내보낸다.

def subsequent_mask(size): # 이해한게 맞나 모르겠다 왜냐면 룩어헤드 마스크 구조의 함수라서... 
    attn_shape = (1, size, size) # 튜플 생성
    subsequent_mask = np.triu(np.ones(attn_shape), k =1).astype('uint8') # 1번째 대각선 아래로 0으로 채우고 나머지 1
    return torch.from_numpy(subsequent_mask) == 0  # torch.from_numpy 텐서로 변환해도 메모리 공유라
                                                    # 텐서 값 바뀌면 마스크 array 값도 바뀜

def attention(query, key, value, mask = None, dropout = None): # 이해한게 맞나 모르겠다
    "Compute 'Scaled Dot Product Attention ' "
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
    / math.sqrt(d_k)                # 주목 점수 구하는 부분
    if mask is not None:            # 마스크 있으면
        scores = scores.masked_fill(mask == 0, -1e9)  # 마스크가 false이면 아주 작은 값으로 채움
    p_attn = F.softmax(scores, dim = -1)     # 소프트맥스 적용
    if dropout is not None:         # 드롭아웃 있으면
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # 어탠션 스코어와 어탠션 매트릭스 리턴

"MultiHeadAttention class"
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        "Take in model size and number of heads"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # 가정설정문을 통한 head로 d_model이 나눠지지 않는 경우
        self.d_k = d_model // h # 64 차원
        self.h = h  # head 개수 8
        self.linears = clones(nn.Linear(d_model, d_model), 4) #4개의 d_model*d_model 선형함수 생성성        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1) # 마스크 차원 1차원위치에 1 증가
        nbatches = query.size(0)  # 쿼리의 첫번째 차원 크기 만큼 배치 개수

        # 1) do all the linear projectionsin batch from d_model => h * d_k
        query, key, value = \
        [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
         for l, x in zip(self.linears, (query, key, value))]

        # 2) apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask = mask, # 어탠션
                                 dropout = self.dropout)

        # 3) concat using a view and apply a final linear
        x = x.tranpose(1, 2).contiguous()\
        .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

## FeedForward class

class PositionwiseFeedForward(nn.Module): # 이해완료
    "Implements FFN equation"
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 가중치 행렬 w1 -> xw + b
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
## is it need?? -> forward 함수 자체는 nn.Module 상속이기 때문에 오버라이드 해서 써야 하긴 함
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
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 사인함수 인코딩
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 코사인 함수 인코딩
        pe = pe.unsqueeze(0)   # 0차원 위치에 1차원 추가(배치 크기 추가해준 것 같음)
        self.register_buffer('pe', pe)  # pe가 학습되지 않도록 함

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
plt.show()
