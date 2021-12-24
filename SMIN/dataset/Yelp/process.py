# coding=utf-8
import json
import pickle
import scipy.sparse as sp
import numpy as np
import datetime
import time
itemJson = "yelp_academic_dataset_business.json"
userJson = "yelp_academic_dataset_user.json"
ratingJson = "yelp_academic_dataset_review.json"

startDate = datetime.datetime.strptime('2018-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
startUTC = time.mktime(startDate.timetuple())
endDate = datetime.datetime.strptime('2020-1-1 00:00:00', '%Y-%m-%d %H:%M:%S')
endUTC = time.mktime(endDate.timetuple())

#business_id categories
def getItemCategory():
    f = open(itemJson, 'r', encoding='utf-8')
    itemCategory = {}
    catSet = set()
    itemName2ID = {}
    itemID = 0
    for line in f.readlines():
        dic = json.loads(line)
        itemStrID = dic['business_id'].strip()
        if dic["categories"]:
            catList = list(map(str.strip, dic["categories"].split(',')))
        else:
            continue
        itemCategory[itemStrID] = catList
        itemName2ID[itemStrID] = itemID
        itemID += 1
        for i in catList:
            catSet.add(i)
    categoryList = list(catSet)
    itemNum = len(itemCategory.keys())
    categoryMat = sp.dok_matrix((itemNum, len(categoryList)),dtype=np.int)
    itemList = list(itemCategory.keys())
    for item in itemList:
        tmp = itemCategory[item]
        iid = itemName2ID[item]
        for category in tmp:
            idx = categoryList.index(category)
            categoryMat[iid, idx] = 1
    with open("category.csv", 'wb') as fs:
        pickle.dump(categoryMat.tocsr(), fs)
    with open("itemName2ID.csv", 'wb') as fs:
        pickle.dump(itemName2ID, fs)

# user_id friends
def getUserFriends():
    f = open(userJson, 'r', encoding='utf-8')
    userFriends = {}
    userID = 0
    userName2ID = {}
    # userNameList = []
    # userIDList = []
    for line in f.readlines():
        dic = json.loads(line)
        userStrID = dic['user_id'].strip()
        if dic["friends"]:
            fList = list(map(str.strip, dic["friends"].split(',')))
        else:
            continue
        userFriends[userStrID] = fList
        userName2ID[userStrID] = userID
        userID += 1
        # userNameList.append(userStrID)
        # userIDList.append(userID)
    userNum = len(userName2ID.keys())
    trust = sp.dok_matrix((userNum, userNum),dtype=np.int)
    for i in list(userName2ID.keys()):
        userName = i
        uid = userName2ID[i]
        friends = userFriends[userName]
        for s in friends:
            if s in userName2ID:
                tid = userName2ID[s]
            else:
                continue
            trust[uid, tid] = 1
    trust = trust.tocsr()
    with open("trust.csv", 'wb') as fs:
        pickle.dump(trust, fs)
    with open("userName2ID.csv", 'wb') as fs:
        pickle.dump(userName2ID, fs)


def getRatings():
    with open("itemName2ID.csv", 'rb') as fs:
        itemName2ID = pickle.load(fs)

    with open("userName2ID.csv", 'rb') as fs:
        userName2ID = pickle.load(fs)
    userNum = len(userName2ID.keys())
    itemNum = len(itemName2ID.keys())
    
    ratingMat = sp.dok_matrix((userNum, itemNum),dtype=np.int)
    timeMat = sp.dok_matrix((userNum, itemNum),dtype=np.int)

    f = open(ratingJson, 'r', encoding='utf-8')
    for line in f.readlines():
        dic = json.loads(line)
        # date = datetime.datetime.strptime(dic['date'].split(' ')[0], '%Y-%m-%d %H:%M:%S')
        tmp_date = datetime.datetime.strptime(dic['date'], '%Y-%m-%d %H:%M:%S')
        tmpUTC = time.mktime(tmp_date.timetuple())
        if tmpUTC > startUTC and tmpUTC < endUTC:
            uidStr = dic['user_id']
            iidStr = dic['business_id']
            if uidStr.strip() in userName2ID and iidStr.strip() in itemName2ID:
                uid = userName2ID[uidStr.strip()]
                iid = itemName2ID[iidStr.strip()]
                rating = dic['stars']
                # date = datetime.datetime.strptime(dic['date'], '%Y-%m-%d %H:%M:%S')
                # utc = time.mktime(date.timetuple())
                ratingMat[uid, iid] = rating
                timeMat[uid, iid] = tmpUTC

    with open('ratings.csv', 'wb') as fs:
        pickle.dump(ratingMat.tocsr(), fs)
    with open('time.csv', 'wb') as fs:
        pickle.dump(timeMat.tocsr(), fs)

if __name__ == "__main__":
    getItemCategory()
    getUserFriends()
    getRatings()
    print("Done")