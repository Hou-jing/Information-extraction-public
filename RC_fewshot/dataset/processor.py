# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 14:21
# @Author  : Hj
# @FileName: processor.py
# @Software: PyCharm
#get the sentence chunk pharse
import argparse
import ast
import json

import nltk
parser=argparse.ArgumentParser()
parser.add_argument('--filename',default='train.txt')
parser.add_argument('--newfilename',default='chunk_train.txt')
args = parser.parse_args()
#--to translate the original sentence to chunk sentence
def get_sentence_chunk(args):
    path='E:\\PythonProject\\pythonProject_22.10.24_few_shot\\小样本想法\\fused_chunk_information for RE\\dataset\\retacred\\k-shot\\16-1\\'#\\k-shot\\16-1
    fp=open(path+args.filename,mode='r',encoding='utf_8').readlines()
    newfp=open(path+args.newfilename,mode='w+',encoding='utf_8')
    for line in fp:
        line=ast.literal_eval(line)
        sent=line['token']
        h_pos,t_pos=line['h']['pos'],line['t']['pos']
        pos=h_pos+t_pos
        h,t=sorted(pos)[1],sorted(pos)[2]
        par_sent=sent[h:t+1]
        sentences = nltk.pos_tag(par_sent)
        grammar = """VP:{<VBG><TO>|<VBN><TO>|<VBD><TO>|<VB><TO>|<VBP><TO>|<VBZ><TO>}
                   VP:{<VBG><IN>|<VBN><IN>|<VBD><IN>|<VB><IN>|<VBP><IN>|<VBZ><IN>}     
                   NP:{<DT>?<RB>*<JJ>*<NN>+|<DT>?<RB>*<JJ>*<NNP>+|<DT>?<RB>*<JJ>*<NNS>+|<DT>?<RB>*<JJ>*<NNPS>+}
                   PP:{<IN><IN>}
                   """
        # 创建一个块解析器
        cp = nltk.RegexpParser(grammar)
        # 测试。查看结果
        result = cp.parse(sentences)
        substring = []
        finalstring = ''
        for subtree in result.subtrees():
            if ((subtree.label() == 'NP') | (subtree.label() == 'PP') | (subtree.label() == 'VP')):
                substring.append(subtree)
        for each in substring:
            length = len(each)
            # for i in (0,length-1):
            # print(each[i])
            for i in range(0, length):
                finalstring += each[i][0] + ' '
            finalstring += ''
        if h_pos[0]<t_pos[0]:
            new_sent='[S]'+line['h']['name']+'[S]'+' '+finalstring+'[O]'+line['t']['name']+'[O]'
            sent.insert(h_pos[0],'[S]')
            sent.insert(h_pos[1]+1,'[S]')
            sent.insert(t_pos[1]+2,'[O]')
            sent.insert(t_pos[1]+3,'[O]')
        else:
            new_sent = '[O]'+line['t']['name']+'[O]'+ ' ' + finalstring +'[S]'+ line['h']['name']+'[S]'
            sent.insert( h_pos[0] +2 ,'[S]')
            sent.insert(h_pos[1] + 3,'[S]' )
            sent.insert( t_pos[1] ,'[O]')
            sent.insert(t_pos[1] + 1,'[O]')
        old_sent=' '.join(sent)
        rel=line['relation']
        d={'new_sent':new_sent,'old_sent':old_sent,'rel':rel}

        newfp.write(json.dumps(d,ensure_ascii=False))
        newfp.write('\n')

get_sentence_chunk(args)