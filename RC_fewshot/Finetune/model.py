# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 8:38
# @Author  : Hj
# @FileName: model_rel_pro.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, AlbertTokenizer, AlbertModel, AutoTokenizer, AutoModel


class BertForSequenceClassification(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_labels = args.class_num
        self.args = args
        if args.embed_mode == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
            self.encoder = AlbertModel.from_pretrained("albert-xxlarge-v1")
        elif args.embed_mode == 'bert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.encoder = AutoModel.from_pretrained("bert-base-cased")


        classifier_dropout = (
            args.dropout if args.dropout is not None else args.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(args.hidden_size, self.num_labels)
        self.device = args.device
        # Initialize weights and apply final processing
    def forward(
        self,
       data,labels
    , return_dict=None) :#labels:bs
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        emb = self.tokenizer(data, return_tensors="pt",
                             max_length=128,
                             padding=True,
                             truncation=True,
                             is_split_into_words=False).to(self.device)
        outputs = self.encoder(
            emb['input_ids'],emb['attention_mask']
        )

        pooled_output = outputs[1]#bs*dim
        '''pool_output是CLS的embedding，hidden state是整个句子的embedding'''

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)#bs*num_labels

        loss = None
        if labels is not None:
            if self.args.problem_type is None:
                if self.num_labels == 1:
                    self.args.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.args.problem_type = "single_label_classification"
                else:
                    self.args.problem_type = "multi_label_classification"

            if self.args.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.args.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.args.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if self.args.type == 'train':
            return loss,logits
        else:
            return logits
