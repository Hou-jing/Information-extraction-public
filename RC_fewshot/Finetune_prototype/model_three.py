import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer
from torch.cuda.amp import autocast



class Pro(nn.Module):
    def __init__(self,args, config):
        super(Pro, self).__init__()
        self.args=args
        self.tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path, config=config)
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        hidden_size = config.hidden_size
    def forward(self):
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
        return sub,obj,rel
    def get_temps(self):
        temps = []
        with open(self.args.data_dir + "/" + self.args.temps, "r+") as f:
            for i in f.readlines():
                i = i.strip().split("\t")
                tar=i[2]+i[3]+ i[4]+i[5]+i[6]
                temps.append(tar)
        return temps








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
        rel_emb=pooled_output[idx,0]
        s=torch.einsum('ij,kj->ik', ss_emb, sub)
        o=torch.einsum('ij,kj->ik', os_emb, obj)
        r=torch.einsum('ij,kj->ik', rel_emb, rel)
        # logits=torch.einsum('ij,kj->ik',ss_emb,sub)+torch.einsum('ij,kj->ik',os_emb,obj)+torch.einsum('ij,kj->ik',rel_emb,rel)
        logits=r
        '''#-----FIXME CLS分类模型-----
        cls_emb=pooled_output[:,0,:]
        logits = self.classifier2(cls_emb)'''
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs
