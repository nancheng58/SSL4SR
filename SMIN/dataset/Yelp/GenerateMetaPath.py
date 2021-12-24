import numpy as np
import scipy.sparse as sp
import pickle
import datetime

print(datetime.datetime.now())

np.random.seed(30)  #make generate the same metapath when running this code

with open("data.pkl", 'rb') as fs:
    data = pickle.load(fs)

(trainMat, _, trustMat, categoryMat, categoryDict) = data

trainMat=trainMat.tocsr()
trustMat = trustMat.tocsr()
categoryMat = categoryMat.tocsr()

userNum, itemNum = trainMat.shape

#UU
mat = (trustMat.T + trustMat) + sp.eye(userNum).tocsr()
UU_mat = (mat != 0)
with open('./UU_mat.csv','wb') as fs:
    pickle.dump(UU_mat,fs)
del UU_mat #释放内存
print('UU Done\n')

#UIU
UIU_mat = sp.dok_matrix((userNum, userNum))
for i in range(userNum):
    data = trainMat[i].toarray()
    _, iid = np.where(data != 0)
    uidList, _ = np.where(np.sum(trainMat[:, iid]!=0, axis=1) != 0)
    uidList2 = np.random.choice(uidList, size=int(uidList.size*0.3), replace=False)
    uidList2 = uidList2.tolist()
    tmp = [i] * len(uidList2)
    UIU_mat[tmp, uidList2] = 1
UIU_mat = UIU_mat.tocsr()
UIU_mat = UIU_mat + UIU_mat.T + sp.eye(userNum).tocsr()
UIU_mat = (UIU_mat != 0)
with open('./UIU_mat.csv','wb') as fs:
    pickle.dump(UIU_mat,fs)
del UIU_mat #释放内存
print('UIU Done\n')

#UITIU
UITIU_mat = sp.dok_matrix((userNum, userNum))
for i in range(userNum):
    data = trainMat[i].toarray()
    _, iid = np.where(data != 0)
    _, typeidList = np.where(np.sum(categoryMat[iid]!=0,axis=0)!=0)
    typeidSet = set(typeidList.tolist())
    for typeid in typeidSet:
        iidList = categoryDict[typeid]
        uidList, _ = np.where(np.sum(trainMat[:, iidList]!=0, axis=1) != 0)
        uidList2 = np.random.choice(uidList, size=int(uidList.size*0.0003), replace=False)
        uidList2 = uidList2.tolist()
        tmp = [i] * len(uidList2)
        UITIU_mat[tmp, uidList2] = 1
UITIU_mat = UITIU_mat.tocsr()
UITIU_mat = UITIU_mat + UITIU_mat.T + sp.eye(userNum).tocsr()
UITIU_mat = (UITIU_mat != 0)
with open('./UITIU_mat.csv','wb') as fs:
    pickle.dump(UITIU_mat,fs)
del UITIU_mat #释放内存
print('UITIU Done\n')

#ITI
ITI_mat = sp.dok_matrix((itemNum, itemNum))
for i in range(categoryMat.shape[0]):
    data = categoryMat[i].toarray()
    _,typeList = np.where(data!=0)
    itemList, _ = np.where(np.sum(categoryMat[:,typeList]!=0,axis=1) != 0)
    itemList2 = np.random.choice(itemList, size=int(itemList.size*0.002), replace=False)
    itemList2 = itemList2.tolist()
    tmp = [i] * len(itemList2)
    ITI_mat[tmp, itemList2] = 1
ITI_mat = ITI_mat.tocsr()
ITI_mat = ITI_mat + ITI_mat.T + sp.eye(itemNum).tocsr()
ITI_mat = (ITI_mat != 0)
with open('./ITI_mat.csv','wb') as fs:
    pickle.dump(ITI_mat,fs)
del ITI_mat
print('ITI Done\n')

#IUI
IUI_mat = sp.dok_matrix((itemNum, itemNum))
trainMat_T = trainMat.T
for i in range(itemNum):
    data = trainMat_T[i].toarray()
    _, uid = np.where(data != 0)
    iidList, _ = np.where(np.sum(trainMat_T[:, uid] != 0, axis=1) != 0)
    iidList2 = np.random.choice(iidList, size=int(iidList.size*0.25), replace=False)
    iidList2 = iidList2.tolist()
    tmp = [i] * len(iidList2)
    IUI_mat[tmp, iidList2] = 1
IUI_mat = IUI_mat.tocsr()
IUI_mat = IUI_mat + IUI_mat.T + sp.eye(itemNum).tocsr()
IUI_mat = (IUI_mat != 0)
with open('./IUI_mat.csv','wb') as fs:
    pickle.dump(IUI_mat,fs)
del IUI_mat
print('IUI Done\n')
print(datetime.datetime.now())


#=====================================
with open('./UU_mat.csv','rb') as fs:
    UU_mat = pickle.load(fs) #2300857
with open('./UIU_mat.csv','rb') as fs:
    UIU_mat = pickle.load(fs) #9768734
with open('./UITIU_mat.csv','rb') as fs:
    UITIU_mat = pickle.load(fs) #11934323
with open('./ITI_mat.csv','rb') as fs:
    ITI_mat = pickle.load(fs) #5796878
with open('./IUI_mat.csv','rb') as fs:
    IUI_mat = pickle.load(fs) #2830870

metaPath = {}
metaPath['UU'] = UU_mat
metaPath['UIU'] = UIU_mat
metaPath['UITIU'] = UITIU_mat
metaPath['ITI'] = ITI_mat
metaPath['IUI'] = IUI_mat

with open('metaPath.pkl', 'wb') as fs:
    pickle.dump(metaPath, fs)
print("Done")
