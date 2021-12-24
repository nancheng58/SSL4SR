# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader  
import torch.optim as optim
import pickle
import random
import numpy as np
import time
import dgl
from dgl import DGLGraph
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import argparse
import os

from ToolScripts.TimeLogger import log
from ToolScripts.tools import sparse_mx_to_torch_sparse_tensor
from Interface.BPRData import BPRData  
import Interface.evaluate as evaluate
from model import MODEL
from MV_MIL.informax import Informax

modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")


isLoadModel = False
LOAD_MODEL_PATH = r"SR-HAN_Yelp_1599990303_hide_dim_8_layer_dim_[8,8,8]_lr_0.05_reg_0.02_topK_10_lambda1_0_lambda2_0"


class Hope():
    def __init__(self,args,data,metaPath,subGraph):
        self.args = args 
        self.metaPath = metaPath

        #train data and test data
        trainMat, testData, _, _, _ = data
        self.userNum, self.itemNum = trainMat.shape
        train_coo = trainMat.tocoo()
        train_u, train_v, train_r = train_coo.row, train_coo.col, train_coo.data
        assert np.sum(train_r == 0) == 0
        train_data = np.hstack((train_u.reshape(-1,1),train_v.reshape(-1,1))).tolist()#//(u,v)list
        test_data = testData
        
        train_dataset = BPRData(train_data, self.itemNum, trainMat, 1, True) #num_negtive samples
        test_dataset = BPRData(test_data, self.itemNum, trainMat, 0, False)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0) 
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=1024*1000, shuffle=False,num_workers=0) #test batch=1024

        #user metaPath: UU UIU UITIU ITI IUI
        self.uu_graph = dgl.graph(self.metaPath['UU'], ntype='user', etype='social')
        self.uiu_graph = dgl.graph(self.metaPath['UIU'], ntype='user', etype='rating')
        self.uitiu_graph = dgl.graph(self.metaPath['UITIU'], ntype='user', etype='rating') 
        # self.user_graph =[self.uu_graph, self.uiu_graph, self.uitiu_graph] #7 cases

        #item metapath IUI ITI
        self.iti_graph = dgl.graph(self.metaPath['ITI'], ntype='item', etype='category')
        self.iui_graph = dgl.graph(self.metaPath['IUI'], ntype='item', etype='raitng')
        # self.item_graph =[self.iui_graph, self.iti_graph] #3 cases
        
        #according args to append metapath graph to user graph or item graph
        self.graph_dict={}
        self.graph_dict['uu']=self.uu_graph
        self.graph_dict['uiu']=self.uiu_graph
        self.graph_dict['uitiu']=self.uitiu_graph
        self.graph_dict['iui']=self.iui_graph
        self.graph_dict['iti']=self.iti_graph

        print("user metaPath: "+self.args.user_graph_indx)
        user_graph_list = self.args.user_graph_indx.split('_')
        self.user_graph = []
        for i in range(len(user_graph_list)):
            self.user_graph.append(self.graph_dict[user_graph_list[i]])

        print("item metaPath: "+self.args.item_graph_indx)
        item_graph_list = self.args.item_graph_indx.split('_')
        self.item_graph = []
        for i in range(len(item_graph_list)):
            self.item_graph.append(self.graph_dict[item_graph_list[i]])
        del self.graph_dict, self.uu_graph, self.uiu_graph, self.uitiu_graph, self.iui_graph, self.iti_graph
        
        #informax
        if self.args.informax == 1:
            (self.ui_graphAdj,self.ui_subGraphAdj) = subGraph
            self.ui_subGraphAdj_Tensor = sparse_mx_to_torch_sparse_tensor(self.ui_subGraphAdj).cuda()
            self.ui_subGraphAdj_Norm =t.from_numpy(np.sum(self.ui_subGraphAdj,axis=1)).float().cuda()
            self.ui_graph = DGLGraph(self.ui_graphAdj)

        #data for plot 
        self.train_losses = []
        self.test_hr = []
        self.test_ndcg = []
    
    def prepareModel(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        self.out_dim = self.args.hide_dim + sum(eval(self.args.layer_dim))
        #metapath encoder model
        self.model = MODEL(len(self.user_graph),
                           len(self.item_graph),
                           self.userNum,
                           self.itemNum,
                           self.args.hide_dim,
                           eval(self.args.layer_dim)).cuda()
        #informax
        if self.args.informax == 1:
            if self.args.informax_graph_act == 'sigmoid':
                informaxGraphAct = nn.Sigmoid()
            elif self.args.informax_graph_act == 'tanh':
                informaxGraphAct = nn.Tanh()
            print('informax graph-level Act funciton: '+self.args.informax_graph_act )
            self.ui_informax = Informax(self.ui_graph,self.out_dim, self.out_dim, nn.PReLU(), informaxGraphAct,self.ui_graphAdj).cuda()
            self.opt = optim.Adam([
                {'params':self.model.parameters(),'weight_decay':0},
                {'params':self.ui_informax.parameters(),'weight_decay':0},
                ],lr=self.args.lr)
        else:
            self.opt = optim.Adam(self.model.parameters(),lr=self.args.lr)

    def predictModel(self,user, pos_i, neg_j, isTest=False):
        if isTest:
            pred_pos = t.sum(user * pos_i, dim=1)
            return pred_pos
        else:
            pred_pos = t.sum(user * pos_i, dim=1)
            pred_neg = t.sum(user * neg_j, dim=1)
            return pred_pos, pred_neg

    def adjust_learning_rate(self):
        # lr = self.lr * (self.args.decay**epoch)
        if self.opt != None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)
                # print(param_group['lr'])

    def getModelName(self):
        title = "SR-HAN" + "_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr +\
        "_hide_dim_" + str(self.args.hide_dim) +\
        "_layer_dim_" + str(self.args.layer_dim) +\
        "_lr_" + str(self.args.lr) +\
        "_reg_" + str(self.args.reg) +\
        "_topK_" + str(self.args.topk) +\
        "_graph_" + str(self.args.user_graph_indx) +"_"+ str(self.args.item_graph_indx) +\
        "_useInformax_" + str(self.args.informax) +\
        "_"+str(self.args.k_hop_num) + "hopSubGraph"+\
        "_lambda1_" + str(self.args.lambda1) +\
        "_lambda2_" + str(self.args.lambda2)
        return ModelName

    def saveHistory(self): 
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        ModelName = self.getModelName()

        with open(r'./History/' + dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self): 
        ModelName = self.getModelName()
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        savePath = r'./Model/' + dataset + r'/' + ModelName + r'.pth'
        params = {
            'model': self.model,
            'epoch': self.curEpoch,
            'args': self.args,
            'opt': self.opt,
            'history':history
            }
        t.save(params, savePath)
        log("save model : " + ModelName)

    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.model = checkpoint['model']
        self.args = checkpoint['args']
        self.opt = checkpoint['opt']

        history = checkpoint['history']
        self.train_losses = history['loss']
        self.test_hr = history['hr']
        self.test_ndcg = history['ndcg']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))

    def trainModel(self):
        epoch_loss = 0
        epoch_informax_loss=0
        self.train_loader.dataset.ng_sample() 
        for user, item_i, item_j in self.train_loader:  
            ##a batch
            bpr_loss = 0

            user = user.long().cuda()  
            item_i =item_i.long().cuda()
            item_j = item_j.long().cuda()
            self.userEmbed,self.itemEmbed = self.model(self.user_graph, self.item_graph)

            #predict
            pred_pos, pred_neg = self.predictModel(self.userEmbed[user], self.itemEmbed[item_i], self.itemEmbed[item_j])
            bprloss = -(pred_pos.view(-1) - pred_neg.view(-1)).sigmoid().log().sum()
            bpr_loss += bprloss
            
            epoch_loss += bpr_loss.item()
            regLoss=(t.norm(self.userEmbed[user])**2+t.norm(self.itemEmbed[item_i])**2+t.norm(self.itemEmbed[item_j])**2)  
            loss = 0.5*(bpr_loss + regLoss*self.args.reg)/self.args.batch

            #DGIloss  
            if self.args.informax == 1:
                ui_informax_loss = 0
                self.allEmbed = t.cat([self.userEmbed,self.itemEmbed],dim=0) 
                if self.args.lambda1 != 0 or self.args.lambda2 != 0:
                    res = self.ui_informax(self.allEmbed, self.ui_subGraphAdj, self.ui_subGraphAdj_Tensor,self.ui_subGraphAdj_Norm)
                    Mask = t.zeros((self.userNum+self.itemNum)).cuda()
                    Mask[user]=1
                    Mask[self.userNum+item_i] = 1
                    Mask[self.userNum+item_j] = 1
                    informax_loss = self.args.lambda1*(((Mask*res[0]).sum()+(Mask*res[1]).sum())/t.sum(Mask))\
                        +self.args.lambda2*(((Mask*res[2]).sum()+(Mask*res[3]).sum())/t.sum(Mask)+res[4])
                    epoch_informax_loss += informax_loss.item()
                    loss = loss + informax_loss 
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return epoch_loss 

    def testModel(self):
        HR=[]
        NDCG=[]
        with t.no_grad():
            self.userEmbed,self.itemEmbed = self.model(self.user_graph, self.item_graph)

            for test_u, test_i in self.test_loader:
                test_u = test_u.long().cuda()
                test_i = test_i.long().cuda()
                pred = self.predictModel(self.userEmbed[test_u], self.itemEmbed[test_i], None, isTest=True)
                batch = int(test_u.cpu().numpy().size/100)
                for i in range(batch):
                    batch_socres=pred[i*100:(i+1)*100].view(-1)
                    _,indices=t.topk(batch_socres,self.args.topk) 
                    tmp_item_i=test_i[i*100:(i+1)*100]
                    recommends=t.take(tmp_item_i,indices).cpu().numpy().tolist()
                    gt_item=tmp_item_i[0].item()
                    HR.append(evaluate.hit(gt_item,recommends))
                    NDCG.append(evaluate.ndcg(gt_item,recommends))
        return np.mean(HR),np.mean(NDCG)

    def run(self):
        self.prepareModel()
        if isLoadModel:
            self.loadModel(LOAD_MODEL_PATH)
            HR,NDCG = self.testModel()
            log("HR@10=%.4f, NDCG@10=%.4f"%(HR, NDCG))
            return 
            
        loss = 0
        self.curEpoch = 0
        best_hr=-1
        best_ndcg=-1
        best_epoch=-1

        wait=0

        for e in range(args.epochs+1):
            self.curEpoch = e
            #train
            log("**************************************************************")
            epoch_loss = self.trainModel()
            self.train_losses.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f"%(e, args.epochs, epoch_loss))

            #test
            HR, NDCG = self.testModel()
            self.test_hr.append(HR)
            self.test_ndcg.append(NDCG)
            log("epoch %d/%d, HR@10=%.4f, NDCG@10=%.4f"%(e, args.epochs, HR, NDCG))

            self.adjust_learning_rate()     
            if HR>best_hr:
                best_hr,best_ndcg,best_epoch=HR,NDCG,e
                wait=0
                self.saveModel()
            else:
                wait+=1
                print('wait=%d'%(wait))
            
            self.saveHistory()
            if wait==self.args.patience:
                log('Early stop! best epoch = %d'%(best_epoch))
                self.loadModel(self.getModelName())
                break

        print("*****************************")
        log("best epoch = %d, HR= %.4f, NDCG=%.4f"% (best_epoch,best_hr,best_ndcg)) 
        print("*****************************")   
        print(self.args)
        log("model name : %s"%(self.getModelName()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR-HAN main.py')
    parser.add_argument('--dataset', type=str, default='CiaoDVD')
    parser.add_argument('--batch', type=int, default=8192, metavar='N', help='input batch size for training')
    parser.add_argument('--seed', type=int, default=29, metavar='int', help='random seed')
    parser.add_argument('--decay', type=float, default=0.97, metavar='LR_decay', help='decay')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate')
    parser.add_argument('--minlr', type=float,default=0.0001)
    parser.add_argument('--reg', type=float, default=0.05) 
    parser.add_argument('--epochs', type=int, default=400, metavar='N', help='number of epochs to train')
    parser.add_argument('--patience', type=int, default=5, metavar='int', help='early stop patience')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--hide_dim', type=int, default=16, metavar='N', help='embedding size')
    parser.add_argument('--layer_dim',nargs='?', default='[16]', help='Output size of every layer') 
    parser.add_argument('--user_graph_indx', nargs=r"?", default="uu_uiu_uitiu", help='user graph')
    parser.add_argument('--item_graph_indx', nargs=r"?", default="iui_iti", help='item graph')
    parser.add_argument('--gcn_act', default='prelu',help='metaPath gcn activation function')
    #informax
    parser.add_argument('--informax', type=int, default=1, help="whether use informax model block")
    parser.add_argument('--informax_graph_act',default='sigmoid',help='informax graph activation function')
    parser.add_argument('--lambda1', type=float, default=0.06, help='weight of loss with informax')
    parser.add_argument('--lambda2', type=float, default=0.002, help='weight of loss with informax')
    parser.add_argument('--k_hop_num',type=int,default=2,help='k-hop of subgraph')

    args = parser.parse_args()
    print(args)
    dataset = args.dataset

    with open(r'dataset/'+args.dataset+'/metaPath.pkl', 'rb') as fs:
        metaPath = pickle.load(fs)
    with open(r'dataset/'+args.dataset+'/data.pkl', 'rb') as fs:
        data = pickle.load(fs)

    subGraphPath=r'dataset/'+args.dataset+'/'+str(args.k_hop_num)+'hop_ui_subGraph.pkl'
    if not os.path.exists(subGraphPath):
        print('please run '+'dataset/'+args.dataset+'/GenerateSubGraph.py first!')
        exit()
    else:   
        with open(subGraphPath,'rb') as fs:
            subGraph = pickle.load(fs)
    hope = Hope(args,data,metaPath,subGraph)

    modelName = hope.getModelName()
    print('ModelName = ' + modelName)    
    hope.run()
   

    

  

