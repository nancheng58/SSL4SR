import copy
import string
import sys, random, os
from collections import defaultdict

from data.statistics_length import length


def loadfile(filename):
    ''' load a file, return a generator. '''
    fp = open(filename, 'r')
    for i, line in enumerate(fp):
        yield line.strip('\r\n')
        if i % 100000 == 0:
            print('loading %s(%s)' % (filename, i), file=sys.stderr)
    fp.close()
    print('load %s succ' % filename, file=sys.stderr)


def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            for i, r, t in ilist:
                f.write(str(u) + '\t' + str(i) + '\t' + str(t) + "\n")


def generate_dataset(filename, pivot=0.7):
    ''' load rating data and split it to training set and test set '''
    trainset_len = 0
    testset_len = 0
    user_train = {}
    user_valid = {}
    user_test = {}
    User = dict()
    fitter_User = dict()
    usermap = dict()
    DATASET = 'ML-1M'
    if not os.path.isdir('./' + DATASET):
        os.mkdir('./' + DATASET)
    train_file = './' + DATASET + '/train.txt'
    valid_file = './' + DATASET + '/valid.txt'
    test_file = './' + DATASET + '/test.txt'
    imap_file = './' + DATASET + '/imap.json'
    umap_file = './' + DATASET + '/umap.json'
    countU = defaultdict(lambda: 0)  # dict which default 0
    countP = defaultdict(lambda: 0)
    for line in loadfile(filename):
        user, item, rating, time = line.split('::')
        # print('user : ' + user)
        userid = int(user)
        itemid = int(item)
        userid -= 1
        itemid -= 1
        countU[userid] += 1
        countP[itemid] += 1
    for line in loadfile(filename):
        user, item, rating, time = line.split('::')
        # print('user : ' + user)
        userid = int(user)
        itemid = int(item)
        userid -= 1
        itemid -= 1
        if countU[userid] < 5 or countP[itemid] < 5:
            continue
        item = str(itemid)
        user = str(userid)
        if user not in usermap:
            usermap[user] = 1
            User[user] = []
        else:
            User[user].append([item, rating, time])
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

    User = filter_Kcore(User)
    itemmap = dict()

    itemid = 0
    for _, ilist in User.items():
        for i in ilist:
            if i[0] not in itemmap:
                itemmap[i[0]] = itemid
                itemid += 1
    print(len(itemmap))
    final_User = dict()
    for u, ilist in User.items():
        index = 0
        final_User[u] = []
        for i in ilist:
            new_ilist = [itemmap[i[0]], i[1], i[2]]
            index += 1
            final_User[u].append(new_ilist)
    User = final_User
    for user in User.keys():
        User[user].sort(key=lambda x: x[2])
    new_user = 0
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
    writetofile(user_train, train_file)
    writetofile(user_valid, valid_file)
    writetofile(user_test, test_file)

    itemmap = dict()
    for _, ilist in fitter_User.items():
        for i in ilist:
            if i[0] not in itemmap:
                itemmap[i[0]] = 1
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


if __name__ == '__main__':
    ratingfile = os.path.join('ratings.dat')
    # ratingfile2 = os.path.join('yelp','test.in')

    generate_dataset(ratingfile)
