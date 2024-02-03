
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch import detach
import utils
import preprocess as pp
from   mutators_weight import *
import step_5_mutators
from random import choice



def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)



## 4. get_attn_pad_mask
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


## 3. PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

## 10.
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

## 9. Decoder

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def seq(input_seq):
    input = [[d.v2i[v] for v in input_seq]]
    return torch.LongTensor(input)


def decode_sequence(input_seq,diversity=1.0):

    enc_inputs = seq(input_seq)
    dec_inputs = np.zeros(( 1, 2))
    dec_inputs[ 0,0] = 1
    dec_inputs = torch.LongTensor(dec_inputs)
    enc_inputs = enc_inputs.cuda()
    dec_inputs = dec_inputs.cuda()
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    output = detach(outputs).cpu().numpy()
    outdata = [d.i2v[i] for i in np.argmax(output, axis=1)]
    return outdata[0]

def function_prefix_start(text,length):
    startindex =[]
    index = text.find('function')
    startindex.append(index)
    start = index
    while(start < length):
        start += 1
        if(text.find('function',start) != startindex[-1] and text.find('function',start) != -1):
            startindex.append(text.find('function',start))
    return startindex

def constructor_prefix_start(text,length):
    startindex =[]
    index = text.find('constructor')
    startindex.append(index)
    start = index
    while(start < length):
        start += 1
        if(text.find('constructor',start) != startindex[-1] and text.find('constructor',start) != -1):
            startindex.append(text.find('constructor',start))
    return startindex




def synthesis(text, gmode='g1', smode='nosample'):

    if (gmode == 'g1'):

        prefix_start = random.randint(maxlen, len(text) - maxlen)
        prefix = text[prefix_start - maxlen:prefix_start]
        generated = ""
        head = text[0: prefix_start]
        tail = text[prefix_start:]
        cut_index = tail.find(';') + 1
        tail = tail[cut_index:]
        num_line = 0
        k = 0
        while (num_line < max_num_line and k <500):
            k = k + 1
            if (smode == 'nosample'):
                next_char = decode_sequence(prefix, 1)
            if (smode == 'sample'):
                next_char = decode_sequence(prefix, 1.2)
            if (smode == 'samplespace'):
                if (generated[-1] == ' ' or generated[-1] == ';'):
                    next_char = decode_sequence(prefix, 1.2)
                else:
                    next_char = decode_sequence(prefix, 1)
            if (next_char == ';'):
                num_line += 1
            if(next_char == '}' and num_line == 1):
                break
            generated += next_char
            prefix = prefix[1:] + next_char
        text = head + generated + tail


    if (gmode == 'g2'):
        for i in range(2):
            prefix_start = random.randint(maxlen, len(text) - maxlen)
            prefix = text[prefix_start - maxlen:prefix_start]
            generated = ""
            head = text[0: prefix_start]
            tail = text[prefix_start:]
            cut_index = tail.find(';') + 1
            tail = tail[cut_index:]
            num_line = 0
            k = 0
            while (num_line < max_num_line / 2 and k < 500):
                k = k + 1
                if (smode == 'nosample'):
                    next_char = decode_sequence(prefix, 1)
                if (smode == 'sample'):
                    next_char = decode_sequence(prefix, 1.2)
                if (smode == 'samplespace'):
                    if (generated[-1] == ' ' or generated[-1] == ';'):
                        next_char = decode_sequence(prefix, 1.2)
                    else:
                        next_char = decode_sequence(prefix, 1)
                if (next_char == ';'):
                    num_line += 1
                generated += next_char
                prefix = prefix[1:] + next_char
            text = head + generated + tail

    if (gmode == 'g3'):

        start_funct_list = function_prefix_start(text, len(text))
        rn_start_funct_list = removenoise_fun(text, start_funct_list, len(text))
        long_list_fun = []
        max_list_fun = []
        for i in rn_start_funct_list:
            startIndex_fun, endIndex_fun = 0, 0
            startIndex_fun, endIndex_fun = Count_length(text, i, len(text))
            long_list_fun.append(startIndex_fun)
            long_list_fun.append(endIndex_fun)
            max_list_fun.append(endIndex_fun - startIndex_fun + 1)
        start_max_const = long_list_fun[(max_list_fun.index(max(max_list_fun))) * 2]
        end_max_const = long_list_fun[(max_list_fun.index(max(max_list_fun))) * 2 + 1]
        prefix_start = random.randint(start_max_const + 1, end_max_const - 2)
        prefix = text[prefix_start - maxlen:prefix_start]
        generated = ""
        head = text[0: prefix_start]
        tail = text[prefix_start:]
        num_chop_line = 0

        while (num_chop_line < max_num_line ):
            cut_index = tail.find(';') + 1
            tail = tail[cut_index:]
            num_chop_line += 1
        num_line = 0
        k = 0
        while (num_line < max_num_line and k < 500):
            k = k + 1
            if (smode == 'nosample'):
                next_char = decode_sequence(prefix, 1)
            if (smode == 'sample'):
                next_char = decode_sequence(prefix, 1.2)
            if (smode == 'samplespace'):
                if (generated[-1] == ' ' or generated[-1] == ';'):
                    next_char = decode_sequence(prefix, 1.2)
                else:
                    next_char = decode_sequence(prefix, 1)
            if (next_char == ';'):
                num_line += 1
            generated += next_char
            prefix = prefix[1:] + next_char
        text = head + generated + tail

    print('-' * 50)
    print('head: ')
    print(head)
    print('generated: ')
    print(generated)
    print(len(generated))
    print('tail: ')
    print(tail)
    return text

def reWrite( text, path):
    f = open(path, 'w')
    f.write(text)
    return 1

def mut(text):
    PROJECT_DIR = os.getcwd()
    MUTATORS_DIR = PROJECT_DIR + "/mutG1_3/"
    mutChoiceList = ["m1", "m2", "m3"]

    if os.path.exists(MUTATORS_DIR) == False:
        os.makedirs(MUTATORS_DIR)

    for i in range(3):
        mutMode = random.choice(mutChoiceList)
        text = mutators_strategy(text, mutMode, file)
    reWrite(text, MUTATORS_DIR + os.path.basename(file))



def generate(i):
    total_count = 0
    files = []
    for root, d_names, f_names in os.walk(seed_path):
        for f in f_names:
            files.append(os.path.join(root, f))
    for file in files:
        try:
            text = open(file, 'r',encoding="utf-8").read()
        except  UnicodeError:
            continue
        total_count += 1

        try:
            text = synthesis(text, 'g1', 'nosample')
            pp.Rewrite(text, file)
        except RuntimeError:
            continue







if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_path = os.getcwd()
    seed_path = root_path + "/Comparison_Model_data"
    maxlen = 50
    max_num_line = 2

    max_decoder_seq_length =2

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    d = utils.DateData()
    src_vocab_size = tgt_vocab_size = len(d.vocab)
    model = Transformer().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    model.load_state_dict(torch.load( root_path + "/model_50/model.pkl"))
    optimizer.load_state_dict(torch.load(root_path + "/model_50/opt.pkl"))
    optimizer.zero_grad()
    generate()





