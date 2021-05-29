import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

from dataset import START, PAD

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, nc, leakyRelu=False):
        super(CNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        def convRelu(i, batchNormalization=False):
            cnn = nn.Sequential()
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
            return cnn

        self.conv0 = convRelu(0)
        self.pooling0 = nn.MaxPool2d(2, 2)
        self.conv1 = convRelu(1)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.conv2 = convRelu(2, True)
        self.conv3 = convRelu(3)
        self.pooling3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv4 = convRelu(4, True)
        self.conv5 = convRelu(5)
        self.pooling5 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv6 = convRelu(6, True)
    
    def forward(self, input):

        out = self.conv0(input)     # [batch size, 64, 128, 128]
        out = self.pooling0(out)    # [batch size, 64, 64, 64]
        out = self.conv1(out)       # [batch size, 128, 64, 64]
        out = self.pooling1(out)    # [batch size, 128, 32, 32]
        out = self.conv2(out)       # [batch size, 256, 32, 32]
        out = self.conv3(out)       # [batch size, 256, 32, 32]
        out = self.pooling3(out)    # [batch size, 256, 16, 33]
        out = self.conv4(out)       # [batch size, 512, 16, 33]
        out = self.conv5(out)       # [batch size, 512, 16, 33]
        out = self.pooling5(out)    # [batch size, 512, 8, 34]
        out = self.conv6(out)       # [batch size, 512, 7, 33]
        return out

class AttentionCell(nn.Module):
    def __init__(self, src_dim, hidden_dim, embedding_dim, num_layers=1, cell_type='LSTM'):
        super(AttentionCell, self).__init__()
        self.num_layers = num_layers

        self.i2h = nn.Linear(src_dim, hidden_dim, bias=False)
        self.h2h = nn.Linear(
            hidden_dim, hidden_dim
        )  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        if num_layers == 1:
            if cell_type == 'LSTM':
                self.rnn = nn.LSTMCell(src_dim + embedding_dim, hidden_dim)
            elif cell_type == 'GRU':
                self.rnn = nn.GRUCell(src_dim + embedding_dim, hidden_dim)
            else:
                raise NotImplementedError
        else:
            if cell_type == 'LSTM':
                self.rnn = nn.ModuleList(
                    [nn.LSTMCell(src_dim + embedding_dim, hidden_dim)]
                    + [
                        nn.LSTMCell(hidden_dim, hidden_dim)
                        for _ in range(num_layers - 1)
                    ]
                )
            elif cell_type == 'GRU':
                self.rnn = nn.ModuleList(
                    [nn.GRUCell(src_dim + embedding_dim, hidden_dim)]
                    + [
                        nn.GRUCell(hidden_dim, hidden_dim)
                        for _ in range(num_layers - 1)
                    ]
                )
            else:
                raise NotImplementedError

        self.hidden_dim = hidden_dim



    def forward(self, prev_hidden, src, tgt):   # src: [batch_size, L, c]
        '''한번의 time step에 대한 forward이다.
        '''

        src_features = self.i2h(src)  # [batch_size, L, h] = (batch_size, 231, 128)
        if self.num_layers == 1:
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)    # [batch_size, 1, h]
        else:
            prev_hidden_proj = self.h2h(prev_hidden[-1][0]).unsqueeze(1)    # [batch_size, 1, h]
        attention_logit = self.score(
            torch.tanh(src_features + prev_hidden_proj) # [batch_size, L, h]
        )  # [batch_size, L, 1]
        alpha = F.softmax(attention_logit, dim=1)  # [batch_size, L, 1]
        context = torch.bmm(alpha.permute(0, 2, 1), src).squeeze(1)  # [batch_size, c]
        '''alpha.permute(0,2,1) shape (batch_size, 1, L)
        src shape [batch_size, L, c]
        
        alpha는 [w_1, w_2, ..., w_L], src는 [C_1, C_2, ..., C_L](각 C는 크기 c의 벡터)이라면,
        context는 (w_1*C_1) + (w_2*C_2) + ...+ (w_L*C_L) 이다.
        '''

        concat_context = torch.cat([context, tgt], 1)  # [batch_size, c+e]

        if self.num_layers == 1:
            cur_hidden = self.rnn(concat_context, prev_hidden)
            '''(Pdb) cur_hidden[0].size()
            torch.Size([26, 128])
            (Pdb) cur_hidden[1].size()
            torch.Size([26, 128])
            왜 cur_hidden이 두개의 텐서로 이루어져 있는지 모르겟음. GRUCell의 doc를 봐도 output은 (batch, hidden dim)의 hidden state 텐서 하나만 나온다고 되어 있는데.
            
            '''
        else:
            cur_hidden = []
            for i, layer in enumerate(self.rnn):
                if i == 0:
                    concat_context = layer(concat_context, prev_hidden[i])
                else:
                    concat_context = layer(concat_context[0], prev_hidden[i])
                cur_hidden.append(concat_context)

        return cur_hidden, alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        src_dim,
        embedding_dim,
        hidden_dim,
        pad_id,
        st_id,
        num_layers=1,
        cell_type='LSTM',
        checkpoint=None,
    ):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.attention_cell = AttentionCell(
            src_dim, hidden_dim, embedding_dim, num_layers, cell_type
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.generator = nn.Linear(hidden_dim, num_classes)
        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(
        self, src, text, is_train=True, teacher_forcing_ratio=1.0, batch_max_length=50
    ):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [START] token. text[:, 0] = [START].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        

        batch_size = src.size(0)
        num_steps = batch_max_length - 1  # -1 for [s] at end of sentence.

        output_hiddens = ( ## hidden state의 place holder. shape (batch_size, num_steps, self.hidden_dim)
            torch.FloatTensor(batch_size, num_steps, self.hidden_dim)
            .fill_(0)
            .to(device)
        )

        ## initial hidden state. 전부 0으로 채우고 있음.
        if self.num_layers == 1:
            hidden = (
                torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
            )
        else:
            hidden = [
                (
                    torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                    torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                )
                for _ in range(self.num_layers)
            ]

        if is_train and random.random() < teacher_forcing_ratio:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                embedd = self.embedding(text[:, i])
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, src, embedd)
                if self.num_layers == 1:
                    output_hiddens[:, i, :] = hidden[
                        0
                    ]  # LSTM hidden index (0: hidden, 1: Cell)
                else:
                    output_hiddens[:, i, :] = hidden[-1][0]
            probs = self.generator(output_hiddens)

        else:
            targets = (
                torch.LongTensor(batch_size).fill_(self.st_id).to(device)
            )  # [START] token 인 0으로 채워진 텐서. shape (batch_size,)
            probs = ( # shape (batch_size, num_steps, self.num_classes). output으로 나오는 logit들을 담을 placeholder
                torch.FloatTensor(batch_size, num_steps, self.num_classes)
                .fill_(0)
                .to(device)
            )

            for i in range(num_steps):
                embedd = self.embedding(targets) ## shape (batch_size, embedding_dim = 128)
                hidden, alpha = self.attention_cell(hidden, src, embedd)
                if self.num_layers == 1:
                    probs_step = self.generator(hidden[0]) ## shape (batch_size, # vocab)
                else:
                    probs_step = self.generator(hidden[-1][0])
                probs[:, i, :] = probs_step ## 해당 time step에서 구한 logit을 probs에 넣는다.
                _, next_input = probs_step.max(1) ## shape(batch_size,)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class Attention(nn.Module):
    def __init__(
        self,
        FLAGS,
        train_dataset,
        checkpoint=None,
    ):
        super(Attention, self).__init__()
        
        self.encoder = CNN(FLAGS.data.rgb)
        
        self.decoder = AttentionDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.Attention.src_dim,
            embedding_dim=FLAGS.Attention.embedding_dim,
            hidden_dim=FLAGS.Attention.hidden_dim,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            num_layers=FLAGS.Attention.layer_num,
            cell_type=FLAGS.Attention.cell_type)

        self.criterion = (  ## 네트워크마다 loss가 달라서 네트워크 선언할 때 loss도 같이 선언해준다.
            nn.CrossEntropyLoss()
        )

        if checkpoint:
            self.load_state_dict(checkpoint)
    
    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        out = self.encoder(input)
        b, c, h, w = out.size()
        out = out.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c]
        output = self.decoder(out, expected, is_train, teacher_forcing_ratio, batch_max_length=expected.size(1))    # [b, sequence length, class size]
        return output