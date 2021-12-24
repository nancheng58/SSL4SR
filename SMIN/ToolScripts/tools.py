import torch as t
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

def showSparseTensor(tensor):
    index = t.nonzero(tensor)
    countArr = t.sum(tensor!=0, dim=1).cpu().numpy()
    start=0
    end=0
    tmp = tensor[index[:,0], index[:,1]].cpu().detach().numpy()
    for i in countArr:
        start = end
        end += i
        print(tmp[start: end])

def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
    exps = t.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    ret = (masked_exps/masked_sums)
    return ret

def list2Str(s):
    ret = str(s[0])
    for i in range(1, len(s)):
        ret = ret + '_' + str(s[i])
    return ret

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()