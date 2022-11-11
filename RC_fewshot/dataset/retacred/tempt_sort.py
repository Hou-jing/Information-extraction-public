# -*- coding: utf-8 -*-
# @Time    : 2022/11/9 21:51
# @Author  : Hj
# @FileName: tempt_sort.py
# @Software: PyCharm

##  --Fixme----将tempt中句子按照rel2id的顺序排序
import json
newfp=open('new_temp.txt','w+',encoding='utf_8')
with open('temp.txt', "r+") as f:
    temps=[]
    for i in f.readlines():
        i = i.strip().split("\t")
        rel = i[1]
        temps.append(rel)
print(temps)
lines=open('temp.txt', "r+").readlines()
rel2id = json.load(open('rel2id.json', encoding='utf_8'))
for j in list(rel2id.keys()):
    txt = temps.index(j)
    newfp.write(lines[txt])



