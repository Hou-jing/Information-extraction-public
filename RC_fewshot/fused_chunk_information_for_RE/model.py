# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 15:13
# @Author  : Hj
# @FileName: model_rel_pro.py
# @Software: PyCharm
#sentence_encode

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification
class Encoder(nn.Module):
    def __init__(self, class_num, args):
        super().__init__()
        self.args = args
        self.class_num = class_num
        self.dropout_value = args.dropout

        # net structures and operations
        self.dropout = nn.Dropout(self.dropout_value)
        self.dense = nn.Linear(
                    in_features=args.dim,
                    out_features=int(args.dim/2)
                    # bias=True
                )#bs*class_num
        self.dense_=nn.Linear(
            in_features=int(args.dim/2),
            out_features=self.class_num,
            bias=True
        )
    def forward(self, reps):
        if self.args.selector == 'avg':
            rep = torch.mean(reps, dim=1, keepdim=False)#bs*dim
        else:
            rep=reps[:, 0, :] #bs*dim
        bag_reps = self.dropout(rep)
        bag_reps =self.dense(bag_reps)
        logits = self.dense_(bag_reps)
        probs = F.softmax(logits, dim=-1)
        return probs


'''class Encoder_seq(nn.Module):
    def __init__(self, class_num, args):
        super().__init__()
        self.args = args
        self.class_num = class_num
        self.model=model = BertForSequenceClassification.from_pretrained("bert-base-cased", config=args)
    def forward(self, reps,label):
        output = self.model(**reps, labels=label)

        # y_pred_prob = logits : [batch_size, num_labels]
        y_pred_prob = output[1].argmax(dim=1)
        
        return output'''


class RC_Model(nn.Module):
    def __init__(self, args,rel2idx):
        super().__init__()
        self.args=args
        self.class_num =len(list(rel2idx.keys()))
        self.max_len = args.max_seq_len
        if args.embed_mode == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
            self.encoder = AlbertModel.from_pretrained("albert-xxlarge-v1")
        elif args.embed_mode == 'bert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.encoder = AutoModel.from_pretrained("bert-base-cased")
        self.logit=Encoder(self.class_num,args)
        self.loss_fnt = nn.CrossEntropyLoss()
        self.loss_dis=nn.MSELoss()
        self.device=args.device
    def forward(self, data, label):
        self.tokenizer.add_tokens(['S','O'])
        emb = self.tokenizer(data, return_tensors="pt",
                           max_length=128,
                           padding=True,
                           truncation=True,
                           is_split_into_words=False).to(self.device)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        reps = self.encoder(emb['input_ids'],emb['attention_mask'])[0]#B*(piece*hidden)
        batch=reps.shape[0]
        probs = self.logit(reps)#bs*class_nums
        suploss=self.loss_fnt(probs,label)
        if self.args.type=='train':

            x,y=probs[0:batch:2,:],probs[1:batch:2,:]#x是原句得到维度，y是chunk句得到的概括
            # kl=F.kl_div(x.softmax(dim=-1), y.softmax(dim=-1), reduction='mean')
            # kl=self.loss_dis(x.softmax(dim=-1), y.softmax(dim=-1))
            # loss = suploss + self.args.alpha_t * kl
            loss = suploss
            # loss2=self.args.alpha_t * kl
            logits=x + y
            return suploss,loss,logits
        else:
            x, y = probs[0:batch:2, :], reps[1:batch:2, :]
            logits = x + y
            return logits

from transformers import BertForSequenceClassification


