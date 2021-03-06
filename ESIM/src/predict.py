import torch

from ESIM import ESIM
from dataset import ValidationDataSet, validation_collater


params = torch.load('params/03-04-18-18-fold0-epoch6.pkl')

epoch_list = [9, 7, 9, 8, 8]  # 5 fold 0.83
path = 'params/03-04-18-18-fold{fold}-epoch{epoch}.pkl'

epoch_list = [7, 5, 6, 6, 6]
path = 'params/03-05-14-11-fold{fold}-epoch{epoch}.pkl'
# epoch_list = [8, 6, 8, 7, 7]
params_list = [torch.load(path.format(fold=i, epoch=epoch))
               for i, epoch in enumerate(epoch_list)]

models = [ESIM(linear_size=50, hidden_size=128)
          for i in range(len(params_list))]
for model, params in zip(models, params_list):
    model.load_state_dict(params)
    model.eval()

folds = len(epoch_list)

dataset = ValidationDataSet('data/gaiic_track3_round1_testA_20210228.tsv')

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, collate_fn=validation_collater)

sigmoid = torch.nn.Sigmoid()

res = []
outputs = []
for i, batch in enumerate(dataloader):
    sent1, sent2 = batch['sentence1'], batch['sentence2']

    preds = []
    for model in models:
        pred = model(sent1, sent2)
        preds.append(sigmoid(pred))
    output = torch.stack(preds, dim=-1).squeeze(1)
    output = output.tolist()

    # voting phase
    # using soft-voting
    for row in output:
        postive_cnt = 0
        postive_sum, negative_sum = 0.0, 0.0  # soft-voting
        postive_max, negative_min = 0.0, 0.0  # hard-voting

        for pred in row:
            if pred >= 0.5:
                postive_cnt += 1
                postive_sum += pred
                postive_max = max(postive_max, pred)
            else:
                negative_sum += pred
                negative_min = min(negative_min, pred)

        # 判断预测结果是正类多还是负类多，然后对预测概率计算均值
        # soft-voting
        if postive_cnt > 2:
            res.append(postive_sum/postive_cnt)  # soft-voting
        else:
            res.append(negative_sum/(folds - postive_cnt))  # soft-voting

    if i % 20 == 19:
        print('%d another 20 finished.' % i)


# outputs = torch.cat(outputs, dim=0).tolist()

f = open('result.txt', mode='w')

for row in res:
    f.write(str(row)+'\n')
    f.flush()

f.close()
