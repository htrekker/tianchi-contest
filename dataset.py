from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd


class TestDataSet(Dataset):

    def __init__(self, file_path, device):
        print('[DataSet] Loading data from %s.' % file_path)
        self.df = pd.read_csv(file_path, header=None, sep='\t')
        print('[DataSet] Total count: %d.' % len(self.df))
        self.device = device

    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = torch.LongTensor(list(map(int, row[0].split())))
        x = x.to(device=self.device)
        y = torch.LongTensor(list(map(int, row[1].split())))
        y = y.to(device=self.device)

        label = float(row[2])

        return {"sentence1": x, "length1": len(x),
                "sentence2": y, "length2": len(y), "label": label,
                "id": index}

    def __len__(self):
        return len(self.df)


def collect_fn(batch_dict):
    batch_size = len(batch_dict)

    sentences1, sentences2 = [], []
    batch_label = []
    id_batch = []
    lens1, lens2 = [], []

    labels = []
    for i in range(batch_size):
        dic = batch_dict[i]
        sentences1.append(dic['sentence1'])
        sentences2.append(dic['sentence2'])

        batch_label.append(dic['label'])
        id_batch.append(dic['id'])

        lens1.append(dic['length1'])
        lens2.append(dic['length2'])

        labels.append(dic['label'])
    res = {}

    res['sentence1'] = pad_sequence(sentences1, batch_first=True)
    res['sentence2'] = pad_sequence(sentences2, batch_first=True)
    res['id'] = id_batch
    res['lengths1'] = lens1
    res['lengths2'] = lens2
    # res['label'] = torch.zeros(batch_size, 2).scatter_(
    #     1, torch.LongTensor(labels).unsqueeze(1), 1)
    res['label'] = torch.FloatTensor(labels)

    return res


if __name__ == '__main__':
    batch_size = 32

    train_set = TestDataSet('data/gaiic_track3_round1_train_20210228.tsv')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, collate_fn=collect_fn)

    for id_x, batch in enumerate(train_loader):
        if id_x > 10:
            break
        print(id_x)
        print(batch)