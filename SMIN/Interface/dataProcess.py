import pickle 
import numpy as np
import scipy.sparse as sp
import random
import os
import argparse
from create_adj import creatMultiItemUserAdj



def splitData(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    with open(DIR + "/category.csv", 'rb') as fs:
        category = pickle.load(fs)
    with open(DIR + "/ratings.csv", 'rb') as fs:
        data = pickle.load(fs)
    with open(DIR + "/times.csv", 'rb') as fs:
        time = pickle.load(fs)
    with open(DIR + "/trust.csv", 'rb') as fs:
        trust = pickle.load(fs)
    assert np.sum(data.tocoo().row != time.tocoo().row) == 0
    assert np.sum(data.tocoo().col != time.tocoo().col) == 0
    row, col = data.shape
    print("user num = %d, item num = %d"%(row, col))

    train_row, train_col, train_data, train_data_time = [], [], [], []
    test_row, test_col, test_data, test_data_time = [], [], [], []

    userList = np.where(np.sum(data!=0, axis=1)>=2)[0]
    for i in userList:
        tmp_data = data[i].toarray()[0]
        if np.sum(tmp_data != 0) < 2:
            continue
        tmp_data_time = time[i].toarray()[0]
        uid = [i] * col 
        num = data[i].nnz
        #降序排序
        idx = np.argsort(-tmp_data_time).tolist()
        idx = idx[: num]
        rating_data = tmp_data[idx].tolist()
        time_data = tmp_data_time[idx].tolist()
        assert np.sum(tmp_data[idx] == 0) == 0
        assert np.sum(tmp_data_time[idx] == 0) == 0
        
        test_num = 1
        train_num = num - test_num

        test_row += uid[0:test_num]
        test_col += idx[0:test_num]
        test_data += rating_data[0:test_num]
        test_data_time += time_data[0:test_num]
        assert (0 in test_data) == False
        assert (0 in test_data_time) == False

        train_row += uid[0:train_num]
        train_col += idx[test_num:]
        train_data += rating_data[test_num:]
        train_data_time += time_data[test_num:]
        assert (0 in train_data) == False
        assert (0 in train_data_time) == False


    train = sp.csc_matrix((train_data, (train_row, train_col)), shape=data.shape)
    test  = sp.csc_matrix((test_data, (test_row, test_col)), shape=data.shape)

    train_time = sp.csc_matrix((train_data_time, (train_row, train_col)), shape=data.shape)
    test_time  = sp.csc_matrix((test_data_time, (test_row, test_col)), shape=data.shape)


    print("train num = %d, train rate = %.2f"%(train.nnz, train.nnz/data.nnz))
    print("test num = %d, test rate = %.2f"%(test.nnz, test.nnz/data.nnz))

    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'wb') as fs:
        pickle.dump(train.tocsr(), fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'wb') as fs:
        pickle.dump(test.tocsr(), fs)
    with open(DIR + "/implicit/cv{0}/trust.csv".format(cv), 'wb') as fs:
        pickle.dump(trust.tocsr(), fs)
    with open(DIR + "/implicit/cv{0}/categoryMat.csv".format(cv), 'wb') as fs:
        pickle.dump(category.tocsr(), fs)

    with open(DIR + "/implicit/cv{0}/train_time.csv".format(cv), 'wb') as fs:
        pickle.dump(train_time.tocsr(), fs)
    with open(DIR + "/implicit/cv{0}/test_time.csv".format(cv), 'wb') as fs:
        pickle.dump(test_time.tocsr(), fs)

def filterData(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    #filter
    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/categoryMat.csv".format(cv), 'rb') as fs:
        category = pickle.load(fs)

    with open(DIR + "/implicit/cv{0}/train_time.csv".format(cv), 'rb') as fs:
        train_time = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test_time.csv".format(cv), 'rb') as fs:
        test_time = pickle.load(fs)

    with open(DIR + "/implicit/cv{0}/trust.csv".format(cv), 'rb') as fs:
        trust = pickle.load(fs)

    trust = trust + trust.transpose()
    trust = (trust != 0)*1

    a = np.sum(np.sum(train != 0, axis=1) ==0)
    b = np.sum(np.sum(train != 0, axis=0) ==0)
    c = np.sum(np.sum(trust, axis=1) == 0)
    while a != 0 or b != 0 or c != 0:
        if a != 0:
            idx, _ = np.where(np.sum(train != 0, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            train_time = train_time[idx]
            test_time = test_time[idx]
            trust = trust[idx][:, idx]
        elif b != 0:
            _, idx = np.where(np.sum(train != 0, axis=0) != 0)
            train = train[:, idx]
            test = test[:, idx]
            train_time = train_time[:, idx]
            test_time = test_time[:, idx]
            category = category[idx]
        elif c != 0:
            idx, _ = np.where(np.sum(trust, axis=1) != 0)
            train = train[idx]
            test = test[idx]
            train_time = train_time[idx]
            test_time = test_time[idx]
            trust = trust[idx][:, idx]
        a = np.sum(np.sum(train != 0, axis=1) ==0)
        b = np.sum(np.sum(train != 0, axis=0) ==0)
        c = np.sum(np.sum(trust, axis=1) == 0)

    nums = train.nnz+test.nnz
    print("train num = %d, train rate = %.2f"%(train.nnz, train.nnz/nums))
    print("test num = %d, test rate = %.2f"%(test.nnz, test.nnz/nums))

    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'wb') as fs:
        pickle.dump(train, fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'wb') as fs:
        pickle.dump(test, fs)
    with open(DIR + "/implicit/cv{0}/train_time.csv".format(cv), 'wb') as fs:
        pickle.dump(train_time, fs)
    with open(DIR + "/implicit/cv{0}/test_time.csv".format(cv), 'wb') as fs:
        pickle.dump(test_time, fs)
    with open(DIR + "/implicit/cv{0}/trust.csv".format(cv), 'wb') as fs:
        pickle.dump(trust, fs)
    with open(DIR + "/implicit/cv{0}/categoryMat.csv".format(cv), 'wb') as fs:
        pickle.dump(category, fs)

def splitAgain(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'rb') as fs:
        test = pickle.load(fs)
    print(train.nnz)
    print(test.nnz)

    with open(DIR + "/implicit/cv{0}/train_time.csv".format(cv), 'rb') as fs:
        train_time = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test_time.csv".format(cv), 'rb') as fs:
        test_time = pickle.load(fs)

    train = train.tolil()
    test = test.tolil()
    train_time = train_time.tolil()
    test_time = test_time.tolil()
    
    idx = np.where(np.sum(test!=0, axis=1).A == 0)[0]
    for i in idx:
        uid = i
        tmp_data = train[i].toarray()[0]
        if np.sum(tmp_data != 0) < 2:
            continue
        num = train[i].nnz
        tmp_data_time = train_time[i].toarray()[0]
        l = np.argsort(-tmp_data_time).tolist()
        l = l[: num]
        test[uid, l[0]] = train[uid, l[0]]
        test_time[uid, l[0]] = train_time[uid, l[0]]
        train[uid, l[0]] = 0
        train_time[uid, l[0]] = 0
    
    train = train.tocsr()
    train_time = train_time.tocsr()
    test = test.tocsr()
    test_time = test_time.tocsr()
    assert  np.sum(train.tocoo().data == 0)==0
    assert  np.sum(test.tocoo().data == 0)==0
    assert  (train+test).nnz == train.nnz+test.nnz

    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'wb') as fs:
        pickle.dump(train, fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'wb') as fs:
        pickle.dump(test, fs)
    with open(DIR + "/implicit/cv{0}/train_time.csv".format(cv), 'wb') as fs:
        pickle.dump(train_time, fs)
    with open(DIR + "/implicit/cv{0}/test_time.csv".format(cv), 'wb') as fs:
        pickle.dump(test_time, fs)


def generateGraph2(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/trust.csv".format(cv), 'rb') as fs:
        trustMat = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/categoryMat.csv".format(cv), 'rb') as fs:
        categoryMat= pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/categoryDict.csv".format(cv), 'rb') as fs:
        categoryDict = pickle.load(fs)
    
    userNum, itemNum =  train.shape
    assert categoryMat.shape[0] == train.shape[1]
    mat = (trustMat.T + trustMat) + sp.eye(userNum)
    UU_mat = (mat != 0)*1

    ITI_mat = sp.dok_matrix((itemNum, itemNum))
    categoryMat = categoryMat.toarray()
    for i in range(categoryMat.shape[0]):
        itemTypeList = np.where(categoryMat[i])[0]
        for itemType in itemTypeList:
            itemList = categoryDict[itemType]
            itemList = np.array(itemList)
            if itemList.size < 100:
                rate = 0.1
            elif itemList.size < 1000:
                rate = 0.01
            else:
                rate = 0.001
            itemList2 = np.random.choice(itemList, size=int(itemList.size*rate/2), replace=False)
            itemList2 = itemList2.tolist()
            tmp = [i for _ in range(len(itemList2))]
            ITI_mat[tmp, itemList2] = 1

    ITI_mat = ITI_mat.tocsr()
    ITI_mat = ITI_mat + ITI_mat.T + sp.eye(itemNum)
    ITI_mat = (ITI_mat != 0)*1

    uu_vv_graph = {}
    uu_vv_graph['UU'] = UU_mat
    uu_vv_graph['II'] = ITI_mat
    with open(DIR + '/implicit/cv{0}/uu_vv_graph.pkl'.format(cv), 'wb') as fs:
        pickle.dump(uu_vv_graph, fs)

def generateGraph(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/trust.csv".format(cv), 'rb') as fs:
        trustMat = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/categoryMat.csv".format(cv), 'rb') as fs:
        categoryMat= pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/categoryDict.csv".format(cv), 'rb') as fs:
        categoryDict = pickle.load(fs)
    
    userNum, itemNum =  train.shape
    assert categoryMat.shape[0] == train.shape[1]
    mat = (trustMat.T + trustMat) + sp.eye(userNum)
    UU_mat = (mat != 0)*1

    ITI_mat = sp.dok_matrix((itemNum, itemNum))
    itemTypeList = categoryMat.toarray().reshape(-1)
    for i in range(itemTypeList.size):
        itemType = itemTypeList[i]
        itemList = categoryDict[itemType]
        itemList = np.array(itemList)
        itemList2 = np.random.choice(itemList, size=int(itemList.size*0.001), replace=False)
        itemList2 = itemList2.tolist()
        tmp = [i for _ in range(len(itemList2))]
        ITI_mat[tmp, itemList2] = 1

    ITI_mat = ITI_mat.tocsr()
    ITI_mat = ITI_mat + ITI_mat.T + sp.eye(itemNum)
    ITI_mat = (ITI_mat != 0)*1

    uu_vv_graph = {}
    uu_vv_graph['UU'] = UU_mat
    uu_vv_graph['II'] = ITI_mat
    with open(DIR + '/implicit/cv{0}/uu_vv_graph.pkl'.format(cv, 'wb')) as fs:
        pickle.dump(uu_vv_graph, fs)
    
    
def createCategoryDict2(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/categoryMat.csv".format(cv), 'rb') as fs:
        category = pickle.load(fs)
    
    assert category.shape[0] == train.shape[1]
    categoryDict = {}
    categoryData = category.toarray()
    for i in range(categoryData.shape[0]):
        iid = i
        typeList = np.where(categoryData[i])[0]
        # typeid = categoryData[i]
        for typeid in typeList:
            if typeid in categoryDict:
                categoryDict[typeid].append(iid)
            else:
                categoryDict[typeid] = [iid]
    with open(DIR + "/implicit/cv{0}/categoryDict.csv".format(cv), 'wb') as fs:
        pickle.dump(categoryDict, fs)

def createCategoryDict(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'rb') as fs:
        test = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/categoryMat.csv".format(cv), 'rb') as fs:
        category = pickle.load(fs)
    
    assert category.shape[0] == train.shape[1]
    categoryDict = {}
    categoryData = category.toarray().reshape(-1)
    for i in range(categoryData.size):
        iid = i
        typeid = categoryData[i]
        if typeid in categoryDict:
            categoryDict[typeid].append(iid)
        else:
            categoryDict[typeid] = [iid]
    with open(DIR + "/implicit/cv{0}/categoryDict.csv".format(cv), 'wb') as fs:
        pickle.dump(categoryDict, fs)

def testNegSample(dataset, cv):
    DIR = os.path.join(os.path.dirname(os.getcwd()), "dataset", dataset)
    #filter
    with open(DIR + "/implicit/cv{0}/train.csv".format(cv), 'rb') as fs:
        train = pickle.load(fs)
    with open(DIR + "/implicit/cv{0}/test.csv".format(cv), 'rb') as fs:
        test = pickle.load(fs)

    train = train.todok()
    test_u = test.tocoo().row
    test_v = test.tocoo().col
    assert test_u.size == test_v.size
    n = test_u.size
    test_data = []
    for i in range(n):
        u = test_u[i]
        v = test_v[i]
        test_data.append([u, v])
        for t in range(99):
            j = np.random.randint(test.shape[1])
            while (u, j) in train or j == v:
                j = np.random.randint(test.shape[1])
            test_data.append([u, j])
    
    with open(DIR + "/implicit/cv{0}/test_data.csv".format(cv), 'wb') as fs:
        pickle.dump(test_data, fs)




if __name__ == '__main__':
    #python splitTrainTestCv --dataset CiaoDVD --rate 0.8
    parser = argparse.ArgumentParser()
    #dataset params
    parser.add_argument('--dataset', type=str, default="Yelp", help="CiaoDVD,Epinions,Douban")
    parser.add_argument('--pos_num', type=int, default=1)
    parser.add_argument('--neg_num', type=int, default=99)
    parser.add_argument('--cv', type=int, default=1, help="1,2,3,4,5")
    args = parser.parse_args()

    dataset = args.dataset+ "_time"
    pos_num = args.pos_num
    neg_num = args.neg_num

    splitData(dataset, args.cv)
    filterData(dataset, args.cv)
    splitAgain(dataset, args.cv)
    filterData(dataset, args.cv)
    testNegSample(dataset, args.cv)
    
    if dataset == "Yelp_time":
        createCategoryDict2(dataset, args.cv)
    else:
        createCategoryDict(dataset, args.cv)
    creatMultiItemUserAdj(dataset, args.cv)
    if dataset == "Yelp_time":
        generateGraph2(dataset, args.cv)
    else:
        generateGraph(dataset, args.cv)

    print("Done")