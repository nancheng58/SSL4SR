import torch
import torch.nn as nn
import math
from MV_MIL.gcn import GCN
from ToolScripts.tools import sparse_mx_to_torch_sparse_tensor
import numpy as np
from dgl.nn.pytorch import GATConv
from dgl import DGLGraph

   
       
class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, activation)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        #///
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_hidden, n_hidden)))
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,node_embedding,graph_embedding, corrupt=False):
        score = torch.sum(node_embedding*graph_embedding,dim=1) 
        
        if corrupt: #negtive case
            res = self.loss(score,torch.zeros_like(score))
        else:       #positive case
            res = self.loss(score,torch.ones_like(score))
        return res


class Informax(nn.Module):
    def __init__(self,g, n_in, n_h, gcnAct, graphAct, graphAdj):
        super(Informax, self).__init__()
        self.encoder = Encoder(g,n_in,n_h,gcnAct)
        self.discriminator = Discriminator(n_h)
        self.graphAct = graphAct
        self.fc = nn.Linear(n_in,n_h)
        graphAdj_coo = graphAdj.tocoo()
        graphAdj_u, graphAdj_v, graphAdj_r = graphAdj_coo.row, graphAdj_coo.col, graphAdj_coo.data
        self.graphAdj_data = np.hstack((graphAdj_u.reshape(-1,1),graphAdj_v.reshape(-1,1))).tolist()#//(u,v)list
        self.graphAdj_data = np.array(self.graphAdj_data)
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, features, subGraphAdj,subGraphAdjTensor,subGraphAdjNorm):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        
        #subgraph global embedding
        #tmp_features = self.fc(features)
        tmp_features = features
        graphEmbeddings = torch.sparse.mm(subGraphAdjTensor, tmp_features) / subGraphAdjNorm  #[n,d]
        graphEmbeddings = self.graphAct(graphEmbeddings)  
        pos_hi_xj_loss = self.discriminator(positive,graphEmbeddings,corrupt=False)
        neg_hi_xj_loss = self.discriminator(negative,graphEmbeddings,corrupt=True)

        #center nodes
        pos_hi_xi_loss = self.discriminator(positive,features,corrupt=False)
        neg_hi_xi_loss = self.discriminator(negative,features,corrupt=True)
        #edge reconstrucut
        tmp =torch.sigmoid(torch.sum(positive[self.graphAdj_data[:,0]]*positive[self.graphAdj_data[:,1]],dim=1))
        adj_rebuilt = self.mse_loss(tmp,torch.ones_like(tmp))/positive.shape[0]
        
        return pos_hi_xj_loss, neg_hi_xj_loss, pos_hi_xi_loss, neg_hi_xi_loss, adj_rebuilt
