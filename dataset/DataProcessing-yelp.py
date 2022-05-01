import gzip
import io
import sys
from collections import defaultdict
from datetime import datetime
import os
import copy
import json
import time
import tqdm

from data.statistics_length import length

countU = defaultdict(lambda: 0)  # dict which default 0
countP = defaultdict(lambda: 0)
line = 0

DATASET = 'yelp'
dataname = './yelp_academic_dataset_review.json'
# dataname = '/home/zfan/BDSC/projects/datasets/newamazon_reviews/{}.json.gz'.format(DATASET)
if not os.path.isdir('./' + DATASET):
    os.mkdir('./' + DATASET)
train_file = './' + DATASET + '/train.txt'
valid_file = './' + DATASET + '/valid.txt'
test_file = './' + DATASET + '/test.txt'
imap_file = './' + DATASET + '/imap.json'
umap_file = './' + DATASET + '/umap.json'

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
fitter_User = dict()
date_max = '2019-12-31 00:00:00'
date_min = '2019-01-01 00:00:00'
data_flie = './yelp_academic_dataset_review.json'
lines = open(data_flie, encoding="utf8").readlines()
for line in tqdm.tqdm(lines):
    l = json.loads(line.strip())
    asin = l['business_id']
    rev = l['user_id']
    # time = l['date']
    date = l['date']
    score = l['stars']
    text = l['text']
    if date < date_min or date > date_max or float(score) <= 0.0:
        continue
    countU[rev] += 1
    countP[asin] += 1

for line in tqdm.tqdm(lines):
    l = json.loads(line.strip())
    # line += 1
    asin = l['business_id']
    rev = l['user_id']
    # time = l['date']
    date = l['date']
    score = l['stars']
    text = l['text']

    if date < date_min or date > date_max or float(score) <= 0.0:
        continue
    if countU[rev] < 5 or countP[asin] < 5:
        continue
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    if rev in usermap:  # a user who prior appear
        userid = usermap[rev]
    else:  # new user
        userid = usernum  # assign id
        usermap[rev] = userid  # save to map
        User[userid] = []  # construct user list
        usernum += 1
    if asin in itemmap:  # ditto
        itemid = itemmap[asin]
    else:
        itemid = itemnum
        itemmap[asin] = itemid
        itemnum += 1
    User[userid].append([itemid, text, timestamp, date, score])  # user list : [(item_id,time),,,,,,]
# sort reviews in User according to time
# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item[0]] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True  # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core=5, item_core=5):  # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item[0]] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items

user_train = {}
user_valid = {}
user_test = {}
new_user = 0

User = filter_Kcore(User)
itemmap = dict()

itemid = 0
for _, ilist in User.items():
    for i in ilist:
        if i[0] not in itemmap:
            itemmap[i[0]] = itemid
            itemid += 1
final_User = dict()
for u, ilist in User.items():
    index = 0
    final_User[u] = []
    for i in ilist:
        new_ilist = [itemmap[i[0]], i[1], i[2], i[3], i[4]]
        index += 1
        final_User[u].append(new_ilist)
User = final_User
for userid in User.keys():
    User[userid].sort(key=lambda x: x[2])
for user in User:
    nfeedback = len(User[user])
    if nfeedback > 50:
        User[user] = User[user][:50]
    if nfeedback < 5:  # if user interaction quantities < 3 , only as train data
        continue
    else:
        fitter_User[new_user] = User[user]
        user_train[new_user] = User[user][:-2]  # [1~n-2]
        user_valid[new_user] = []
        user_valid[new_user].append(User[user][-2])  # last 2 as valid
        user_test[new_user] = []
        user_test[new_user].append(User[user][-1])  # last one as test
        new_user += 1
length(fitter_User)
print(usernum, itemnum)

def writetofile(data, dfile):
    # print("write :")
    count = 0
    with open(dfile, 'w', encoding='utf-8') as f:
        for u, ilist in sorted(data.items()):
            count += 1
            # print("proceed : "+float(count)/len(data))
            for i, text, t, reviewTime, rating in ilist:
                f.write(str(u) + '\t' + str(i) + '\t' + str(t) + "\n")
                # f.write(str(u) + '\t'+ str(i) + '\t' + str(t) + '\t'+str(rating)+'\t' +"\""+ str(text)+"\""+ '\t' +"\""+ str(reviewTime) +"\""+"\n")


item_fitter = 0
for item in itemmap:
    if itemmap[item] < 5:
        item_fitter += 1
print("item_fitter", item_fitter)
maxlen = 0
minlen = 1000000
avglen = 0
for _, ilist in fitter_User.items():
    listlen = len(ilist)
    maxlen = max(maxlen, listlen)
    minlen = min(minlen, listlen)
    avglen += listlen
avglen /= len(user_train)
print('max length: ', maxlen)
print('min length: ', minlen)
print('avg length: ', avglen)
num_instances = sum([len(ilist) for _, ilist in fitter_User.items()])
print('total user: ', len(user_train))
print('total instances: ', num_instances)
print('total items: ', len(itemmap))
print('density: ', num_instances / (len(fitter_User) * len(itemmap)))
print('valid #users: ', len(user_valid))
numvalid_instances = sum([len(ilist) for _, ilist in user_valid.items()])
print('valid instances: ', numvalid_instances)
numtest_instances = sum([len(ilist) for _, ilist in user_test.items()])
print('test #users: ', len(user_test))
print('test instances: ', numtest_instances)
writetofile(user_train, train_file)
writetofile(user_valid, valid_file)
writetofile(user_test, test_file)
