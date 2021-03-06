from torch import nn
import torch
import torch.nn.functional as F

import gensim


class ESIM(nn.Module):
    def __init__(self, linear_size, hidden_size):
        super(ESIM, self).__init__()
        self.linear_size = linear_size
        self.dropout = 0.5
        self.hidden_size = hidden_size

        # load pre-trained word2vec model
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            '../embeddings/woed2vec_embeddings.kv')

        weights = torch.FloatTensor(word_vectors.wv.vectors)
        self.emb_size = weights.size(1)
        self.embeds = nn.Embedding.from_pretrained(weights)
        print(self.embeds)
        self.bn_embeds = nn.BatchNorm1d(self.emb_size)

        self.lstm1 = nn.LSTM(self.emb_size, self.hidden_size,
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size*8, self.hidden_size,
                             batch_first=True, bidirectional=True)

        print('[Model] Model parameters: \n\t\
        Hidden_size: %d, Embeding size: %d, Dropout rate: %.3f.' % (
            self.hidden_size, self.emb_size, self.dropout))

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, 1),
            # nn.Sigmoid()
            # nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len1 * seq_len2
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len1 * seq_len2
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(
            1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, *input):
        # batch_size * seq_len
        sent1, sent2 = input[0], input[1]
        lens1, lens2 = input[2], input[3]
        mask1, mask2 = sent1.eq(0), sent2.eq(0)
        # mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(
            1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(
            1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * emb_size => batch_size * seq_len * hidden_size
        x1 = nn.utils.rnn.pack_padded_sequence(
            x1, lens1, batch_first=True, enforce_sorted=False)
        x2 = nn.utils.rnn.pack_padded_sequence(
            x2, lens2, batch_first=True, enforce_sorted=False)
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)
        o1, _ = nn.utils.rnn.pad_packed_sequence(
            o1, batch_first=True)
        o2, _ = nn.utils.rnn.pad_packed_sequence(
            o2, batch_first=True)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # Input: batch_size * seq_len * (2 * hidden_size) x 2
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_combined = nn.utils.rnn.pack_padded_sequence(
            q1_combined, lens1, batch_first=True, enforce_sorted=False)
        q2_combined = nn.utils.rnn.pack_padded_sequence(
            q2_combined, lens2, batch_first=True, enforce_sorted=False)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
        q1_compose, _ = nn.utils.rnn.pad_packed_sequence(
            q1_compose, batch_first=True)
        q2_compose, _ = nn.utils.rnn.pad_packed_sequence(
            q2_compose, batch_first=True)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity
