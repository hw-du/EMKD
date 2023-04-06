#This program yields the processed dataset and the attributes for ml-1m.
#user starts from 0
#item starts from 1
import pickle
import json
train, val, test, umap, smap= {}, {}, {}, {}, {}
user_seq={}
item_appear = [0 for _ in range(3953)]

with open("movies.dat",'r',encoding="ISO-8859-1") as f:
    data=f.readlines()
    attr_set=set()
    item2attr={}
    for index,seq in enumerate(data):
        item,name,attr=seq.strip().split("::")
        attr = set(attr.split("|"))
        attr_set=attr_set|attr
        item2attr[int(item)]=attr
    attr_set = list(attr_set)
    for k,v in item2attr.items():
        temp=[]
        for attr in item2attr[k]:
            temp.append(attr_set.index(attr)+1)
        temp.sort()
        item2attr[k]=temp

with open("ratings.dat",'r') as f:
    data=f.readlines()
    for index,seq in enumerate(data):
        user,item,rating,timestamp=map(int,seq.strip().split("::"))
        umap[user]=user-1
        smap[item]=item
        temp = user_seq.get(user-1,[])
        temp.append((item,timestamp))
        user_seq[user-1] = temp
    k = list(smap.keys())
    for index,item in enumerate(k,1):
        item_appear[item]=index

    for i in range(len(user_seq)):
        
        user_seq[i].sort(key = lambda x: x[1])
        items = [item_appear[x[0]] for x in user_seq[i]]
        train[i], val[i], test[i] = items[:-2], items[-2:-1], items[-1:]
    smap={}
    for i in range(max(item_appear)):
        smap[i+1]=[i+1]
    dataset = {'train': train,'val': val,'test': test,'umap': umap,'smap': smap}
    with open("dataset.pkl","wb") as f2:
        pickle.dump(dataset, f2)
    real_item2attr = {}
    for k,v in item2attr.items():
        real_item2attr[str(item_appear[k])]=v
    with open ("ML-1M_item2attributes.json","w") as f3:
        json.dump(real_item2attr,f3)
    

        
