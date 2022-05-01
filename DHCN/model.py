import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from numba import jit


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
        #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
        #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0)
        return item_embeddings


class LineConv(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        # session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        # session_emb_lgcn = torch.sum(session1, 0)
        session_emb_lgcn = np.sum(session, 0)
        return session_emb_lgcn


class DHCN(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset,use_HG, emb_size=100, batch_size=100):
        super(DHCN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.use_HG = use_HG
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(200, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset)
        self.LineGraph = LineConv(self.layers, self.batch_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask):
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
        sess_emb_hgnn = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)
        session_emb_lg = self.LineGraph(self.embedding.weight, D, A, session_item, session_len)
        con_loss = self.SSL(sess_emb_hgnn, session_emb_lg)
        if self.use_HG is True:
            sess_emb = sess_emb_hgnn
        else:
            sess_emb = session_emb_lg
        return item_embeddings_hg, sess_emb, self.beta * con_loss
# rewrite


def samples(candidates, session_seq, targetid, all_item, test_num=99, sample_type ='random'):
    test_samples = []
    test_samples.append(targetid.tolist())
    while len(test_samples) <= test_num:
        if sample_type == 'random':
            sample_ids = np.random.choice(all_item, test_num, replace=False)
        else:  # sample_type == 'pop':
            sample_ids = np.random.choice(all_item, test_num, replace=False, p=0.1)
        sample_ids = [int(item) for item in sample_ids if int(item) not in session_seq and int(item) not in test_samples]
        test_samples.extend(sample_ids)
    test_samples = test_samples[:test_num+1]
    candidates = candidates.tolist()
    filter_candidates = []
    ids = []
    for i in range(len(test_samples)):
        filter_candidates.append(candidates[test_samples[i]])
        ids.append(test_samples[i])
    return np.array(filter_candidates), ids

# @jit(nopython=True)
def find_k_largest(K, candidates, items, targetid, all_items):
    n_candidates = []
    candidates,real_ids = samples(candidates, items, targetid, all_items)
    # candidates_list = candidates.tolist()
    # ids = np.argpartition(candidates_list, -K)[-K:].tolist()
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid

    predict_ids = []
    for id in ids:
        predict_ids.append(real_ids[id])
    return predict_ids#,k_largest_scores


def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, length = data.get_slice(i)
    items = session_item
    A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, sess_emb, con_loss = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask)
    scores = torch.mm(sess_emb, torch.transpose(item_emb_hg, 1, 0))
    return tar, scores, con_loss, items, length

def train_test(model, train_data, test_data,epoch):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)  # batch slices index
    for i in slices:
        model.zero_grad()
        targets, scores, con_loss, _, _ = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['recall%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    dir = './model/MLmodelpara.pth'
    state = {'net':model.state_dict(), 'optimizer':model.optimizer.state_dict(), 'epoch':epoch}
    torch.save(state, dir)
    slices = test_data.generate_batch(model.batch_size)
    n_node = test_data.n_node
    all_items = [i for i in range(0, n_node)]
    all_index = []
    all_tar = []
    all_length = []
    for i in slices:
        tar, scores, con_loss, items, length = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd], items[idd], tar[idd], all_items))
        all_index.extend(index)
        all_tar.extend(trans_to_cpu(tar).detach().tolist())
        all_length.extend(length)
        tar = trans_to_cpu(tar).detach().numpy()
        index = np.array(index)
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                metrics['recall%d' % K].append(len(set(prediction) & set([target])) / float(len(set([target]))))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / np.log2(np.where(prediction == target)[0][0] + 2))
    test_length(all_index, all_tar, all_length)
    return metrics, total_loss


def test_length(index, tar, length):
    torch.autograd.set_detect_anomaly(True)
    top_K = [5, 10, 20]
    # tar = trans_to_cpu(tar).detach().tolist()
    print('start predicting: ', datetime.datetime.now())
    length_lower_bound = [0, 20, 30, 40]
    length_upper_bound = [20, 30, 40, 51]
    for j in range(len(length_lower_bound)):
        print(j)
        metrics = {}
        filter_index = []
        filter_tar = []
        for K in top_K:
            metrics['hit%d' % K] = []
            metrics['recall%d' % K] = []
            metrics['mrr%d' % K] = []
            metrics['ndcg%d' % K] = []
        for i in range(len(length)):  # length filter
            length_seq = length[i]
            if length_lower_bound[j] <= length_seq < length_upper_bound[j]:
                filter_index.append(index[i])
                filter_tar.append(tar[i])
        filter_index = np.array(filter_index)
        filter_tar = np.array(filter_tar)
        for K in top_K:
            for prediction, target in zip(filter_index[:, :K], filter_tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                metrics['recall%d' % K].append(len(set(prediction) & set([target])) / float(len(set([target]))))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / np.log2(np.where(prediction == target)[0][0] + 2))
        print_length_metric(metrics)


def print_length_metric(metrics):
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0, 0]
    for K in top_K:
        metrics['hit%d' % K] = np.mean(metrics['hit%d' % K])
        metrics['recall%d' % K] = np.mean(metrics['recall%d' % K])
        metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K])
        metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K])
        if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
            best_results['metric%d' % K][0] = metrics['hit%d' % K]
        if best_results['metric%d' % K][1] < metrics['recall%d' % K]:
            best_results['metric%d' % K][1] = metrics['recall%d' % K]
        if best_results['metric%d' % K][2] < metrics['mrr%d' % K]:
            best_results['metric%d' % K][2] = metrics['mrr%d' % K]
        if best_results['metric%d' % K][3] < metrics['ndcg%d' % K]:
            best_results['metric%d' % K][3] = metrics['ndcg%d' % K]
    # print(metrics)
    for K in top_K:
        print('Recall@%d: %.4f\tNDCG%d: %.4f\tMRR%d: %.4f\tHit@%d: %.4f\tEpoch: %d,  %d' %
              (K, best_results['metric%d' % K][1], K, best_results['metric%d' % K][3], K,
               best_results['metric%d' % K][2], K, best_results['metric%d' % K][0],
               best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
        with open("yelp" + ".txt", 'a') as f:
            f.write('Recall@%d: %.4f\tNDCG%d: %.4f\tMRR%d: %.4f\tHit@%d: %.4f\tEpoch: %d,  %d' %
                    (K, best_results['metric%d' % K][1], K, best_results['metric%d' % K][3], K,
                     best_results['metric%d' % K][2], K, best_results['metric%d' % K][0],
                     best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]) + '\n')

