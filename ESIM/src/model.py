import torch
import torch.nn as nn
import torch.nn.functional as F

import gensim


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, device, rnn_type='lstm'):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.device = device
        if rnn_type == 'lstm':
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=1, batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)

        model = gensim.models.Word2Vec.load('./word_embeddings')
        weights = torch.FloatTensor(model.wv.vectors)
        self.word_embeddings = nn.Embedding.from_pretrained(weights)

        self.dense = nn.Linear(in_features=hidden_size*2,
                               out_features=hidden_size)

    def forward(self, sent1, lens1, sent2, lens2):

        # batch_size, seq_len, emb_size
        emb_sent1, emb_sent2 = self.word_embeddings(
            sent1), self.word_embeddings(sent2)
        packed_sent1, packed_sent2 = nn.utils.rnn.pack_padded_sequence(
            emb_sent1, lens1, batch_first=True, enforce_sorted=False),
        nn.utils.rnn.pack_padded_sequence(
            emb_sent2, lens2, batch_first=True, enforce_sorted=False)

        # batch_size, seq_len, embeds_dim => batch_size, seq_len, 2*hidden_size
        sent1_output, _ = self.rnn(packed_sent1)
        sent2_output, _ = self.rnn(packed_sent2)

        sent1_output, _ = nn.utils.rnn.pad_packed_sequence(
            sent1_output, batch_first=True)
        sent2_output, _ = nn.utils.rnn.pad_packed_sequence(
            sent2_output, batch_first=True)

        sent1_output = self.dense(sent1_output)
        sent2_output = self.dense(sent2_output)

        # batch_size, seq_len, hidden_size
        return sent1_output, sent2_output


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def sub_mul(self, x, alignd):
        mul, sub = x * alignd, x - alignd
        return torch.cat([mul, sub], dim=-1)

    def forward(self, x1, x2):
        mask1, mask2 = x1.eq(0), x2.eq(0)
        mask1 = mask1.float().masked_fill_(mask1, float('-inf')).unsqueeze(1)
        mask2 = mask2.float().masked_fill_(mask2, float('-inf')).unsqueeze(1)
        # weights: [batch_size, seq_len1, seq_len2]
        weights = torch.bmm(x1, x2.transpose(1, 2))

        # alignd_1: [batch_size, seq_len1, hidden_size]
        alignd1 = torch.bmm(
            F.softmax(weights + mask1.unsqueeze(1), dim=-1), x2)
        # aligned2: [batch_size, seq_len2, hidden_size]
        alignd2 = torch.bmm(
            F.softmax(weights.transpose(1, 2) + mask2.unsqueeze(1), dim=-1), x1)

        return torch.cat([x1, alignd1, self.sub_mul(x1, alignd1)], dim=-1),
        torch.cat([x2, alignd2, self.sub_mul(x2, alignd2)], dim=-1)


class InferenceCompestion(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type='lstm'):
        super(InferenceCompestion, self).__init__()

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, comps1, lens1, comps2, lens2):
        # batch_size, seq_len, emb_size
        packed_comps1, packed_comps2 = nn.utils.rnn.pack_padded_sequence(
            comps1, lens1, batch_first=True, enforce_sorted=False),
        nn.utils.rnn.pack_padded_sequence(
            comps2, lens2, batch_first=True, enforce_sorted=False)
        sent1_output, _ = self.rnn(packed_comps1)
        sent2_output, _ = self.rnn(packed_comps2)

        sent1_output, _ = nn.utils.rnn.pad_packed_sequence(
            sent1_output, batch_first=True)
        sent2_output, _ = nn.utils.rnn.pad_packed_sequence(
            sent2_output, batch_first=True)

        q1_rep = self.apply_multiple(sent1_output)
        q2_rep = self.apply_multiple(sent2_output)

        x = torch.cat([q1_rep, q2_rep], dim=-1)
        return self.fc(x)


class ESIM(nn.Module):

    def __init__(self, bilstm, attention, compestion):
        super(ESIM, self).__init__()
        self.first = bilstm
        self.second = attention
        self.final = compestion

    def forward(self, sent1, lens1, sent2, lens2):
        sent1_o, sent2_o = self.first(sent1, lens1, sent2, lens2)
        q1_align, q2_align = self.attention(sent1_o, lens1, sent2_o, lens2)

        return self.final(q1_align, q2_align)
