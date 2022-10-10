import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-Y','I-Y', 'I-Z',
            'I-B', 'B-Z', 'B-B')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
bert_model = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model)

class NerDataset(Dataset):
    ''' Generate our dataset '''

    def __init__(self, f_path):
        self.sents = []
        self.tags_li = []

        with open(f_path, 'r', encoding='utf-8') as f:
            lines = [line.split('\n')[0] for line in f.readlines()]

        tags = [line.split('\t')[1] for line in lines]
        words = [line.split('\t')[0] for line in lines]
        MAX_LEN = 128 - 2
        word, tag = [], []
        for char, t in zip(words, tags):
            if char!= '.':
                word.append(char)
                tag.append(t)
            else:
                if len(word) > MAX_LEN:
                    self.sents.append(['[CLS]'] + word[:MAX_LEN] + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                else:
                    self.sents.append(['[CLS]'] + word + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] + tag + ['[SEP]'])
                word, tag = [], []

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        token_ids = tokenizer.convert_tokens_to_ids(words)
        laebl_ids = [tag2idx[tag] for tag in tags]
        seqlen = len(laebl_ids)
        return token_ids, laebl_ids, seqlen

    def __len__(self):
        return len(self.sents)


def PadBatch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask


import torch
from transformers import BertModel
from torchcrf import CRF


class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        '''self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=2, bidirectional=True, batch_first=True)'''
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(embedding_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_features(self, sentence,mask):
        '''with torch.no_grad():'''
        enc = self.bert(sentence,mask)[0]
        '''enc, _ = self.lstm(embeds)'''
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence,mask)
        if not is_test:  # Training，return loss
            loss = -self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else:  # Testing，return decoding
            decode = self.crf.decode(emissions, mask)
            return decode




def train(e, model, iterator, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for i, batch in enumerate(iterator):
        step += 1
        x, y, z = batch
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)

        loss = model(x, y, z)
        losses += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(e, losses / (step+1e-3)))


def validate(e, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            step += 1

            x, y, z = batch
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            y_hat = model(x, y, z, is_test=True)

            loss = model(x, y, z)
            losses += loss.item()
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            mask = (z == 1)
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean() * 100

    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(e, losses / step, acc))
    return model, losses / step, acc


def test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, y, z = batch
            x = x.to(device)
            z = z.to(device)
            y_hat = model(x, y, z, is_test=True)
            # Save prediction
            for j in y_hat:
                Y_hat.extend(j)
            # Save labels
            mask = (z == 1).cpu()
            y_orig = torch.masked_select(y, mask)
            Y.append(y_orig)

    Y = torch.cat(Y, dim=0).numpy()
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]

    return y_true, y_pred


if __name__ == "__main__":

    labels = ['O', 'B-Y','I-Y', 'I-Z',
            'I-B', 'B-Z', 'B-B']

    best_model = None
    _best_val_loss = 1e18
    _best_val_acc = 1e-18

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--trainset", type=str, default="data/train.txt")
    parser.add_argument("--validset", type=str, default="data/valid.txt")
    parser.add_argument("--testset", type=str, default="data/test.txt")

    ner = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Bert_BiLSTM_CRF(tag2idx).cuda()

    print('Initial model Done.')
    train_dataset = NerDataset(ner.trainset)
    eval_dataset = NerDataset(ner.validset)
    test_dataset = NerDataset(ner.testset)
    print('Load Data Done.')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=ner.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=PadBatch)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=(ner.batch_size) // 2,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=PadBatch)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(ner.batch_size) // 2,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=PadBatch)

    # optimizer = optim.Adam(self.model.parameters(), lr=ner.lr, weight_decay=0.01)
    optimizer = AdamW(model.parameters(), lr=ner.lr, eps=1e-6)

    # Warmup
    len_dataset = len(train_dataset)
    epoch = ner.n_epochs
    batch_size = ner.batch_size
    total_steps = (len_dataset // batch_size) * epoch if len_dataset % batch_size == 0 else (
                                                                                                        len_dataset // batch_size + 1) * epoch

    warm_up_ratio = 0.1  # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)

    print('Start Train...,')
    for epoch in range(1, ner.n_epochs + 1):

        train(epoch, model, train_iter, optimizer, scheduler, device)
        candidate_model, loss, acc = validate(epoch, model, eval_iter, device)

        if loss < _best_val_loss and acc > _best_val_acc:
            best_model = candidate_model
            _best_val_loss = loss
            _best_val_acc = acc

        print("=============================================")

    y_test, y_pred = test(best_model, test_iter, device)
    print('y_test',y_pred)
    print(metrics.classification_report(y_test, y_pred, labels=labels, digits=3))