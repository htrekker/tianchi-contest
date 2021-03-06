import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
from sklearn import metrics

# from model import ESIM, BiLSTM, Attention, InferenceCompestion
from ESIM import ESIM
from dataset import collect_fn
from dataset import TestDataSet

import argparse
from datetime import datetime


def parse_arg():
    parser = argparse.ArgumentParser('Training hyper-parameters.')

    parser.add_argument('--hidden_size', type=int, default=60)
    parser.add_argument('--emb_size', type=int, default=50)
    parser.add_argument('--linear_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, nargs='?', default=128,
                        help='default is 128')
    parser.add_argument('--lr', type=float, nargs='?', default=1e-2,
                        help='optimizer\'s learning rate (default is 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='optimizer\'s momentum (default 0.9)')
    parser.add_argument('--weight_decay', type=float, default=2e-4,
                        help='optimizer weight decay')
    parser.add_argument('--epoch', type=int, help='Number of Epoch.')
    parser.add_argument('--cuda', default=-1, type=int, required=False,
                        help='use cuda or not (default no)')

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":

    args = parse_arg()
    batch_size = args['batch_size']
    epochs = args['epoch']
    lr = args['lr']
    momentum = args['momentum']
    weight_decay = args['weight_decay']
    hidden_size = args['hidden_size']
    emb_size = args['emb_size']
    linear_size = args['linear_size']
    print('[Trainer] Trainer hyper-parameters: \n\t \
        batch size: %d, lr: %.4f, momentum: %.4f, epochs: %d.' %
          (batch_size, lr, momentum, epochs))

    if args['cuda'] == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    print('[Trainer] Using device: %s.' % device)

    dataset = TestDataSet('../../data/shuffled_dataset.tsv', device=device)

    # init summary writer
    writer = SummaryWriter(log_dir='../log')

    now = datetime.strftime(datetime.now(), format='%m-%d-%H-%M')

    k_folds = 5
    kfold = KFold(n_splits=k_folds)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('[Trainer] Start training fold: %d' % fold)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collect_fn,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collect_fn,
                                                  sampler=test_sampler)

        # bilstm = BiLSTM(input_size=100, hidden_size=hidden_size, device=device)
        # attention = Attention()
        # compestion = InferenceCompestion(
        #     input_size=50, hidden_size=hidden_size)
        # model = ESIM(bilstm, attention, compestion)

        model = ESIM(linear_size=linear_size, hidden_size=hidden_size)
        model = model.to(device=device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1)
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.BCEWithLogitsLoss()

        global_step_cnt = 0
        global_test_step = 0
        for epoch in range(epochs):
            running_loss, running_auc = 0.0, 0.0

            train_losses, test_losses = [], []
            train_aucs, test_aucs = [], []

            # model training step
            model.train()
            for batch_ids, batch in enumerate(train_loader):
                global_step_cnt += 1
                sent1, sent2 = batch['sentence1'], batch['sentence2']
                lens1, lens2 = batch['lengths1'], batch['lengths2']

                label = batch['label'].to(device)
                optimizer.zero_grad()
                output = model(sent1, sent2, lens1, lens2)
                output = output.squeeze(-1)
                loss = criterion(output, label)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

                pred = output.to('cpu').detach()
                label = label.to('cpu')
                # 以下3行全是错的
                pred = pred.detach().numpy()
                auc_score = metrics.roc_auc_score(label, pred)
                running_auc += auc_score

                writer.add_scalar('fold{}_loss/train'.format(fold), scalar_value=loss.item(),
                                  global_step=global_step_cnt)
                writer.add_scalar('fold{}_auc/train'.format(fold), scalar_value=auc_score,
                                  global_step=global_step_cnt)
                train_losses.append(loss.item())
                train_aucs.append(auc_score)

                count_num = 200
                if batch_ids % count_num == count_num - 1:  # print every 2000 iteration
                    print('[Trainer] (Epoch: %d, Iter: %d) Train loss: %.4f, Train auc: %.4f.' %
                          (epoch, batch_ids, running_loss/count_num, running_auc/count_num))
                    running_loss, running_auc = 0.0, 0.0

            # 每个epoch记得平均loss
            writer.add_scalar('fold{}_train/loss'.format(fold),
                              scalar_value=(sum(train_losses)/len(train_losses)), global_step=epoch)
            writer.add_scalar('fold{}_train/auc'.format(fold),
                              scalar_value=(sum(train_aucs)/len(train_aucs)), global_step=epoch)

            # save model after every epoch
            if epoch > 3:
                file_path = '../params/{time}-fold{num}-epoch{epoch}.pkl'.format(
                    time=now, num=fold, epoch=epoch)
                torch.save(model.state_dict(), file_path)

            running_loss, running_auc = 0.0, 0.0

            # model evaluation step
            model.eval()
            with torch.no_grad():
                test_counter = 0
                for test_idx, test in enumerate(test_loader):
                    test_counter += 1
                    global_test_step += 1
                    sent1, sent2 = test['sentence1'], test['sentence2']
                    lens1, lens2 = test['lengths1'], test['lengths2']

                    label = test['label'].to(device)

                    test_output = model(sent1, sent2, lens1, lens2).squeeze()
                    loss = criterion(test_output, label)

                    pred = test_output.to('cpu').detach()
                    label = label.to('cpu')
                    auc_score = metrics.roc_auc_score(label, pred)

                    running_auc += auc_score

                    running_loss += loss.item()
                    writer.add_scalar('fold{}_loss/test'.format(fold),
                                      scalar_value=loss.item(), global_step=global_test_step)
                    writer.add_scalar('fold{}_auc/test'.format(fold),
                                      scalar_value=auc_score, global_step=global_test_step)
                    test_losses.append(loss.item())
                    test_aucs.append(auc_score)

            scheduler.step()

            writer.add_scalar('fold{}_test/loss'.format(fold),
                              scalar_value=(sum(test_losses)/len(test_losses)), global_step=epoch)
            writer.add_scalar('fold{}_test/auc'.format(fold),
                              scalar_value=(sum(test_aucs)/len(test_aucs)), global_step=epoch)

            print('Epoch %d: Train avg loss: %.4f, Train avg auc: %.4f; Test avg loss: %.4f, Test avg auc: %.4f.' %
                  (epoch, sum(train_losses)/len(train_losses), sum(train_aucs)/len(train_aucs),
                   sum(test_losses)/len(test_losses), sum(test_aucs)/len(test_aucs)))
