# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 17:20
# @Author  : Hj
# @FileName: main.py
# @Software: PyCharm
import argparse
import json
import os

import torch
import logging

from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from utils import set_seed
from utils import dataloader,get_f1
from model import BertForSequenceClassification
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def evaluate(test_batch, args):
    steps, test_loss = 0, 0
    key,predict=[],[]
    with torch.no_grad():
        for data in test_batch:
            steps += 1
            text = data[0]
            label = torch.tensor(data[1]).to(device)
            loss, logits = model(text, label)
            test_loss += loss.item()
            key.extend(label.cpu())
            predict.extend(torch.argmax(F.softmax(logits, dim=-1),1).cpu())
    key=key
    predict=predict
    accu, recall, f_score=get_f1(key,predict,args)
    logger.info("------ {} Results ------".format('test or dev'))
    logger.info("loss : {:.4f}".format(test_loss / steps))
    logger.info(
        "rel: p={:.4f}, r={:.4f}, f={:.4f}".format(accu, recall, f_score))
    return accu, recall, f_score

class save_results(object):
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)

        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)

    def save(self, info):
        with open(self.filename, 'a') as out:
            print(info, file=out)
set_seed(0)
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--max_seq_len',default=128)
    parser.add_argument('--shot',default='16-1')
    parser.add_argument('--data',default='retacred')

    parser.add_argument('--dropout',default=0.1)
    parser.add_argument('--hidden_dropout_prob',default=0.2)
    parser.add_argument('--selector',default='cls',choices=['avg','cls'],help='the vector used to predict the label')
    parser.add_argument('--hidden_size',default=4096)
    parser.add_argument('--embed_mode',default='albert',choices=['bert_cased','albert'])
    parser.add_argument('--type',default='train',choices=['train','test'])
    parser.add_argument('--do_eval',default=True,choices=[True,False])
    parser.add_argument('--steps',default=50,help='show result every 50 steps')
    parser.add_argument('--eval_metric',default='micro')
    parser.add_argument('--batch_size',default=12)
    parser.add_argument('--eval_batch_size',default=12)
    parser.add_argument("--output_file", default="test", type=str, required=False,
                        help="name of result file")
    parser.add_argument("--lr", default=1e-5, type=float,  # 1e-5
                        help="initial learning rate,${0.00002 (for most datasets) /0.00001 (for SCIERC)}")
    parser.add_argument('--epoch',default=80)
    parser.add_argument('--alpha_t',default=1,help='the weight of the kl part')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument("--weight_decay", default=False, type=bool,
                        help="weight decaying rate")
    parser.add_argument("--weight_decay_value", default=0.1, type=float,
                        help="weight decaying rate")
    parser.add_argument('--class_num',default=40)
    parser.add_argument('--problem_type',default=None)

    args = parser.parse_args()
    with open("E:/PythonProject/pythonProject_22.10.24_few_shot/灏忔牱鏈兂娉?dataset/" + args.data + "/rel2id.json", "r") as f:
        rel2idx = json.load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_batch, test_batch, dev_batch = dataloader(args)

    if args.type=='train':
        logger.info("------Training------")

    model = BertForSequenceClassification(args)
    model.to(device)
    output_dir = "save/" + args.output_file

    os.makedirs(output_dir,exist_ok=True)
    saved_file = save_results(output_dir + "/" + args.output_file + ".txt", header="# epoch \t train_loss \t  dev_loss \t test_loss \t  dev_rel  \t test_rel")
    model_file = args.output_file + ".pt"
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_result=0
    accu, recall, f_score=None,None,None
    if args.weight_decay:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)

    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_value)


    for epoch in range(args.epoch):
        steps, train_loss = 0, 0
        model.train()
        for data in tqdm(train_batch):
            steps += 1
            optimizer.zero_grad()

            text = data[0]
            label = torch.tensor(data[1]).to(device)
            sup_loss, logits=model(text,label)

            # kl_loss.backward(retain_graph=True)
            sup_loss.backward(retain_graph=True)
            train_loss += sup_loss.item()
            optimizer.step()
            # scheduler.step()
            if steps % args.steps == 0:
                logger.info("Epoch: {}, step: {} / {}, loss = {:.4f},sup_loss = {:.4f},learning_rate={}".format
                            (epoch, steps, len(train_batch), train_loss / steps,sup_loss, optimizer.param_groups[0]['lr']))

        logger.info("------ Training Set Results ------")
        logger.info("loss : {:.4f}".format(train_loss / steps))


        if epoch!=0 and epoch%5==0 and args.do_eval:
            model.eval()
            logger.info("------ Evaluating ------")
            accu, recall, f_score = evaluate(test_batch, args)
            if epoch == 0 or f_score > best_result:
                best_result = f_score
                accu, recall, f_score=accu, recall, f_score
                torch.save(model.state_dict(), output_dir + "/" + model_file)
                logger.info("Best result on dev saved!!!")

            saved_file.save(
                "{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} ".format(epoch, train_loss / steps,
                                                                                                  accu, recall, f_score))

    saved_file.save("best test result : accu: {:.4f} \t recall: {:.4f} \t f_score: {:.4f} ".format(accu, recall, f_score))


