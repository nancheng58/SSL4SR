import numpy as np
import scipy.sparse as sp
import pickle
import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

#generate k-hop subgraph of u-i interaction data
print('*********************************')
k_hop=int(input('input the k value of k-hop subgraph:'))
print("get %d_hop subgraph..."%(k_hop))

np.random.seed(30)  
print(datetime.datetime.now())

with open("data.pkl", 'rb') as fs:
    data = pickle.load(fs)

trainMat, _, _, _, _ = data

userNum = trainMat.shape[0]
itemNum = trainMat.shape[1]
uiNum = userNum + itemNum

#k-hop UI-subgraph 
"""k-hop ui-subgraph"""
ui_subGraph = sp.dok_matrix((uiNum,uiNum))
trainMat_coo = trainMat.tocoo() 
u_list, v_list = trainMat_coo.row, trainMat_coo.col
ui_subGraph[u_list,userNum+v_list] = 1
ui_subGraph[userNum+v_list,u_list] = 1
ui_subGraph =  ui_subGraph.tocsr()
with open('ui_Graph.pkl','wb') as fs:
    pickle.dump(ui_subGraph,fs) #原始ui图

with open('ui_Graph.pkl','rb') as fs:
    ui_mat = pickle.load(fs)
if k_hop > 1:
    for i in tqdm(range(uiNum)):
        data = ui_mat[i].toarray()
        _,idList = np.where(data!=0)
        tmp = k_hop-1
        while tmp > 0:
            _,idList = np.where(np.sum(ui_mat[idList,:],axis=0) >= 5)
            ui_subGraph[i,idList] = 1
            tmp = tmp -1
ui_subGraph = (ui_subGraph !=0)
with open('ui_subGraph.pkl','wb') as fs:
    pickle.dump(ui_subGraph, fs) #k-hop图


with open('ui_Graph.pkl','rb') as fs:
    ui_Graph = pickle.load(fs)
with open('ui_subGraph.pkl','rb') as fs:
    ui_subGraph = pickle.load(fs)
data = (ui_Graph, ui_subGraph)
with open(str(k_hop)+'hop_ui_subGraph.pkl','wb') as fs:
    pickle.dump(data, fs)

print('Done!')
print(datetime.datetime.now())
print('*********************************')










