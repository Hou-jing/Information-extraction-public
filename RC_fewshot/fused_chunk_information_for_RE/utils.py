import ast

import torch
import random
import numpy as np
import json
from transformers import AlbertTokenizer, AutoTokenizer
from torch.utils.data import Dataset,DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_f1(key, prediction,args):
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score,recall_score
    if args.eval_metric=='macro':
        f_score=f1_score(key,prediction,average='macro')
        accu=accuracy_score(key,prediction,normalize=True)
        recall=recall_score(key,prediction,average='macro')
    else:
        f_score=f1_score(key,prediction,average='micro')
        accu = accuracy_score(key, prediction, normalize=True)
        recall=recall_score(key,prediction,average='micro')
    return accu,recall,f_score

def json_loads(path, name,rel2idx):
    file = path + "/" + name

    data = []
    with open(file,'r',encoding='utf_8') as f:
        fp=f.readlines()
        random.shuffle(fp)
        for line in fp:
            a = ast.literal_eval(line)
            data.append([a['old_sent'],rel2idx[a['rel']]])
            data.append([a['new_sent'],rel2idx[a['rel']]])
    return data

class dataprocess(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        words = self.data[idx][0]
        ner_labels = self.data[idx][1]
        return (words,ner_labels)

class collater():
    def __init__(self):
        pass
    def __call__(self, data):
        words = [item[0] for item in data]
        labels = [item[1] for item in data]
        return [words,labels]

def dataloader(args):
    path = "dataset/" + args.data
    with open("dataset/" + args.data + "/rel2id.json", "r") as f:
        rel2idx = json.load(f)
    # if args.data == "retacred":
    train_data = json_loads(path + '/' + 'k-shot/' + args.shot + '/', "chunk_train.txt",rel2idx)
    test_data = json_loads(path, "chunk_test.txt",rel2idx)
    dev_data = json_loads(path, 'chunk_val.txt',rel2idx)

    train_dataset = dataprocess(train_data)
    test_dataset = dataprocess(test_data)
    dev_dataset = dataprocess(dev_data)
    collate_fn = collater()

    train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             collate_fn=collate_fn)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True,
                            collate_fn=collate_fn)
    dev_batch = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True,
                           collate_fn=collate_fn)

    return train_batch, test_batch, dev_batch





