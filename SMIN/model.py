import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
import numpy as np


class MODEL(nn.Module):
    def __init__(self,userMetaPathNum,itemMetaPathNum,userNum, itemNum, hide_dim, layer,activation='prelu'):
        super(MODEL, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hide_dim = hide_dim
        self.layer = [hide_dim] + layer
        self.embedding_dict = self.init_weight(userNum, itemNum, hide_dim)#///随机初始化user和item的embedding
        self.userMetaPathNum = userMetaPathNum
        self.itemMetaPathNum = itemMetaPathNum
        self.in_size = np.sum(self.layer)
        if activation == "leakyrelu":
            self.act = t.nn.LeakyReLU(negative_slope=0.5)
            print("GCN Act funcction:leakyrelu")
        elif activation == "relu":
            self.act = t.nn.ReLU()
            print("GCN Act funcction:relu")
        elif activation == "prelu":
            self.act = t.nn.PReLU()
            print("GCN Act funcction:prelu")
        
        self.userMetaLayers = nn.ModuleList()
        for _ in range(userMetaPathNum):
            userLayers=nn.ModuleList()
            for i in range(0,len(self.layer)-1):
                userLayers.append(GraphConv(self.layer[i],self.layer[i+1], bias=False, activation=self.act))
            self.userMetaLayers.append(userLayers)

        self.itemMetaLayers = nn.ModuleList()
        for _ in range(itemMetaPathNum):
            itemLayers=nn.ModuleList()
            for i in range(0,len(self.layer)-1):
                itemLayers.append(GraphConv(self.layer[i],self.layer[i+1],bias=False,activation=self.act))
            self.itemMetaLayers.append(itemLayers)
        #attention 
        self.semanticUserAttention = SemanticAttention(self.in_size) 
        self.semanticItemAttention = SemanticAttention(self.in_size) 
    
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    
    def forward(self, userGraph, itemGraph, norm=1):
        self.semanticUserEmbeddings =[]
        self.semanticItemEmbeddings =[]
        # if len(self.layer) == 1:
        #     return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']
        
        pathNum,blockNum=np.shape(self.userMetaLayers)
        for i in range(pathNum):
            self.all_user_embeddings = [self.embedding_dict['user_emb']]

            layers=self.userMetaLayers[i]
            for j in range(blockNum):
                layer=layers[j]
                if j==0:   
                    userEmbeddings=layer(userGraph[i],self.embedding_dict['user_emb'])
                else:
                    userEmbeddings=layer(userGraph[i],userEmbeddings)

                if norm == 1:
                    norm_embeddings = F.normalize(userEmbeddings, p=2, dim=1)
                    self.all_user_embeddings += [norm_embeddings]
                else:
                    self.all_user_embeddings += [userEmbeddings]
            self.userEmbedding = t.cat(self.all_user_embeddings,1)  #24 [8,8,8]
            self.semanticUserEmbeddings.append(self.userEmbedding)

        pathNum,blockNum=np.shape(self.itemMetaLayers)
        for i in range(pathNum):
            self.all_item_embeddings = [self.embedding_dict['item_emb']]

            layers=self.itemMetaLayers[i]
            for j in range(blockNum):
                layer=layers[j]
                if j==0:   
                    itemEmbeddings=layer(itemGraph[i],self.embedding_dict['item_emb'])
                else:
                    itemEmbeddings=layer(itemGraph[i],itemEmbeddings)

                if norm == 1:
                    norm_embeddings = F.normalize(itemEmbeddings, p=2, dim=1)
                    self.all_item_embeddings += [norm_embeddings]
                else:
                    self.all_item_embeddings += [itemEmbeddings]
            self.itemEmbedding = t.cat(self.all_item_embeddings,1)  
            self.semanticItemEmbeddings.append(self.itemEmbedding)  
            
        
        self.semanticUserEmbeddings = t.stack(self.semanticUserEmbeddings,dim=1)
        self.semanticItemEmbeddings = t.stack(self.semanticItemEmbeddings,dim=1)  #[item_num,num_path,hidden_dim]
        #attention merge
        self.userEmbed = self.semanticUserAttention(self.semanticUserEmbeddings) 
        self.itemEmbed = self.semanticItemAttention(self.semanticItemEmbeddings) #[item_num,hidden_dim]
        return self.userEmbed, self.itemEmbed


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_dim=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  #[item_num,path_num,1]--mean(0)-->[path_num,1]
        beta = t.softmax(w,dim=0)    #[path_num,1] score
        beta = beta.expand((z.shape[0],)+beta.shape) #[item_num,path_num,1]
        return (beta*z).sum(1)       #[item_num,path_num,hidden_dim]--sum(1)-->[item_num,hidden_dim]


