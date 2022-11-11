import ast


import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel,AutoTokenizer
from torch.cuda.amp import autocast
from prepro import TACREDProcessor

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Pro(nn.Module):
    def __init__(self,args, config):
        super(Pro, self).__init__()
        self.args=args
        self.tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path, config=config)
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        hidden_size = config.hidden_size
        self.LABEL_TO_ID = {'NA': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3,
                            'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6,
                            'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9,
                            'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12,
                            'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15,
                            'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19,
                            'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23,
                            'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27,
                            'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30,
                            'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33,
                            'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36,
                            'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39,
                            'org:dissolved': 40, 'per:country_of_death': 41}
    def forward(self):
        if self.args.prop=='temp':
            temps=self.get_temps()
            inputs=self.tokenizer(temps,max_length=128,
                               padding=True,
                               truncation=True,add_special_tokens=False,return_tensors='pt')
            inputs=inputs.to(self.args.device)
            outputs = self.encoder(
                **inputs
            )

            sub=outputs[0][:,0,:]
            obj=outputs[0][:,-1,:]
            rel=outputs[0][:,2,:]

            ## --TODO---rel proto  通过temp文件得到关系原型
            rel=outputs[0].mean(1)
            return sub,obj,rel
        else:
            ## ----TODO ----rel proto 通过数据集得到的原型
            supports=self.get_suuport()
            keys=sorted(list(supports.keys()))
            ss,oo,rel=[],[],[]
            key=0
            sentss = supports[key]['sents']
            '''ids=supports[key]['input_ids']
            max_len=max([len(i) for i in ids])'''
            labs = supports[key]['labels']
            ss = supports[key]['ss']
            os = supports[key]['os']
            '''mask=self.get_atten_mask(ids)
            ids=self.get_ids(ids)
            print(mask.device)
            print(ids.device)
            mask=mask.resize(len(ids),max_len)#shots*max_len
            ids=ids.resize(len(ids),max_len)'''
            inputs = self.tokenizer(sentss, max_length=128,
                                    padding=True,
                                    truncation=True, add_special_tokens=False, return_tensors='pt')
            inputs = inputs.to(self.args.device)
            outputs = self.encoder(
                **inputs
            )
            for key in keys:
                sentss=supports[key]['sents']
                '''ids=supports[key]['input_ids']
                max_len=max([len(i) for i in ids])'''
                labs=supports[key]['labels']
                ss=supports[key]['ss']
                os=supports[key]['os']
                '''mask=self.get_atten_mask(ids)
                ids=self.get_ids(ids)
                print(mask.device)
                print(ids.device)
                mask=mask.resize(len(ids),max_len)#shots*max_len
                ids=ids.resize(len(ids),max_len)'''
                inputs = self.tokenizer(sentss, max_length=128,
                                        padding=True,
                                        truncation=True, add_special_tokens=False, return_tensors='pt')
                inputs = inputs.to(self.args.device)
                outputs = self.encoder(
                    **inputs
                )

                pooled_output = outputs[0]
                shots=pooled_output.shape[0]
                idx=torch.arange(shots).to(self.args.device)
                ss_emb = pooled_output[idx, ss].mean(0).unsqueeze(0)  # bs*d
                os_emb = pooled_output[idx, os].mean(0).unsqueeze(0)
                rel_emb = pooled_output[idx, 0].mean(0).unsqueeze(0)
                ss.append(ss_emb),os.append(os_emb),rel.append(rel_emb)

            ss_ = torch.tensor([item.cpu().detach().numpy() for item in ss]).cuda()
            so_=torch.tensor([item.cpu().detach().numpy() for item in oo]).cuda()
            rel_=torch.tensor([item.cpu().detach().numpy() for item in rel]).cuda()
            return ss_,so_,rel_

    def get_ids(self,ids):
        max_ids = [len(i) for i in ids]
        max_len = max(max_ids)
        idss = []
        for id in ids:
            id_ = id + [0] * (max_len - len(id))
            idss.append(torch.tensor(id_))
        id_inp = torch.tensor([i.cpu().detach().numpy() for i in idss]).cuda()
        return id_inp
    def get_atten_mask(self,ids):
        len_ids = [len(i) for i in ids]
        max_len = max(len_ids)
        attention_mask = torch.zeros(len(ids), max_len)
        for j in range(len(len_ids)):
            attention_mask[j, :len_ids[j]] = 1
        attention_mask=attention_mask.cuda()
        return attention_mask



    def get_temps(self):
        temps = []
        with open(self.args.data_dir + "/" + self.args.temps, "r+") as f:
            for i in f.readlines():
                i = i.strip().split("\t")
                tar=i[2]+i[3]+ i[4]+i[5]+i[6]
                temps.append(tar)
        return temps

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        """
        Implement the following input formats:
            - entity_mask: [SUBJ-NER], [OBJ-NER].
            - entity_marker: [E1] subject [/E1], [E2] object [/E2].
            - entity_marker_punct: @ subject @, # object #.
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        sents = []
        input_format = self.args.input_format
        if input_format == 'entity_mask':
            subj_type = '[SUBJ-{}]'.format(subj_type)
            obj_type = '[OBJ-{}]'.format(obj_type)
            for token in (subj_type, obj_type):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker':
            subj_start = '[SUBJ-{}]'.format(subj_type)
            subj_end = '[/SUBJ-{}]'.format(subj_type)
            obj_start = '[OBJ-{}]'.format(obj_type)
            obj_end = '[/OBJ-{}]'.format(obj_type)
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker_punct':
            subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
            obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if input_format == 'entity_mask':
                if ss <= i_t <= se or os <= i_t <= oe:
                    tokens_wordpiece = []
                    if i_t == ss:
                        new_ss = len(sents)
                        tokens_wordpiece = [subj_type]
                    if i_t == os:
                        new_os = len(sents)
                        tokens_wordpiece = [obj_type]

            elif input_format == 'entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['[E1]'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['[/E1]']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['[E2]'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['[/E2]']

            elif input_format == 'entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['#'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['#']

            elif input_format == 'typed_entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [subj_start] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + [subj_end]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            elif input_format == 'typed_entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            sents.extend(tokens_wordpiece)
        sents = sents[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

        return input_ids, sents,new_ss + 1, new_os + 1
    def get_suuport(self):
        supports={}
        import os
        train_file =os.path.join(self.args.data_dir, self.args.train_file)
        processor = TACREDProcessor(self.args, self.tokenizer)
        train_features = processor.read(train_file)
        with open(train_file, "r") as fh:
            data = fh.readlines()
        for line in tqdm(data):
            d = ast.literal_eval(line)
            ss, se = d['h']['pos'][0], d['h']['pos'][1]
            os, oe = d['t']['pos'][0], d['t']['pos'][1]
            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]
            input_ids, sents,new_ss, new_os = self.tokenize(tokens, 'sub_type', 'obj_type', ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relation']]
            feature = {
                'sents': sents,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }
            if rel not  in list(supports.keys()):
                supports[rel]={
                    'sents': [],
                    'labels': [],
                    'ss': [],
                    'os': [],
                }
            supports[rel]['sents'].append(' '.join(sents))
            supports[rel]['labels'].append(rel)
            supports[rel]['ss'].append(new_ss)
            supports[rel]['os'].append(new_os)
        return supports




class REModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )
        self.proto=Pro(args, config)
        self.classifier2=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )
    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        sub,obj,rel=self.proto()#rel_num*d
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        '''#-----FIXME 论文模型------
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)'''
        ##------TODO prototype 模型------
        ss_emb = pooled_output[idx, ss]#bs*d
        os_emb = pooled_output[idx, os]
        # rel_emb=pooled_output[idx,0]
        rel_emb=pooled_output.mean(1)
        s=torch.einsum('ij,kj->ik', ss_emb, sub)
        o=torch.einsum('ij,kj->ik', os_emb, obj)
        ## -------TODO dot product-------
        r=torch.einsum('ij,kj->ik', rel_emb, rel)
        # logits=torch.einsum('ij,kj->ik',ss_emb,sub)+torch.einsum('ij,kj->ik',os_emb,obj)+torch.einsum('ij,kj->ik',rel_emb,rel)
        logits=r
        '''#-----FIXME CLS分类模型-----
        cls_emb=pooled_output[:,0,:]
        logits = self.classifier2(cls_emb)'''
        ##-----todo-rel_cDIST--dev_f1 12.3694--test_f1 10.50503-
        r = torch.cdist(rel_emb, rel, p=2)
        ## ----TODO---all_cDIST------
        r =torch.cdist(rel_emb, rel,p=2)
        s = torch.cdist(ss_emb, sub,p=2)
        o = torch.cdist(os_emb, obj,p=2)
        logits = r+s*o
        ## -----TODo ----连乘
        logits = r * s * o
        ##----TODO ----连乘2-- dev_f1 14.46955-- test_f1 12.86148---
        e=torch.cdist(ss_emb+os_emb,sub+obj,p=2)
        logits =r+e


        ## --TODO----乘法计算---dev_f1: 1.3895321908290876-
        #logits=r*e

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs
