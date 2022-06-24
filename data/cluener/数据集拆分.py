import json

with open('dev.json',mode='r+',encoding='utf_8') as fp:
    lis=[]
    for line in fp:
        line=json.loads(line)
        lis.append(line)
    dev_data=lis[:-500]
    test_data=lis[-500:]
    print(dev_data[:2])
    with open('dev_pro.json', mode='w+', encoding='utf_8') as f:
        for i in dev_data:
            json.dump(i,f,ensure_ascii=False)
            f.write('\n')
    with open('test_pro.json', mode='w+', encoding='utf_8') as f:
        step=0
        for i in test_data:
            i['id']=step
            json.dump(i,f,ensure_ascii=False)
            f.write('\n')
            step+=1