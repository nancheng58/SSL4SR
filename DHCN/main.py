import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.02, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--use_HG', type=bool, default=True, help='use LG or HG as session emb, default: HG')

opt = parser.parse_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# torch.cuda.set_device(1)

def main():
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    if opt.dataset == 'Beauty':
        n_node = 12101
    elif opt.dataset == 'ML-1M':
        n_node = 3414
    elif opt.dataset == 'yelp':
        n_node = 16552
    else:  # Tmall
        n_node = 40727
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=True, n_node=n_node)
    model = trans_to_cuda(DHCN(adjacency=train_data.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset,use_HG=opt.use_HG))

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0,0,0]
        best_results['metric%d' % K] = [0, 0,0,0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data,epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K])
            metrics['recall%d' % K] = np.mean(metrics['recall%d' % K])
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K])
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K])
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['recall%d' % K]:
                best_results['metric%d' % K][1] = metrics['recall%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][2] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][2] = epoch
            if best_results['metric%d' % K][3] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][3] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][3] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tNDCG%d: %.4f\tMRR%d: %.4f\tHit@%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][1], K, best_results['metric%d' % K][3], K,
                   best_results['metric%d' % K][2], K, best_results['metric%d' % K][0],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            with open(opt.dataset + ".txt", 'a') as f:
                f.write('train_loss:\t%.4f\tRecall@%d: %.4f\tNDCG%d: %.4f\tMRR%d: %.4f\tHit@%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][1], K, best_results['metric%d' % K][3], K,
                   best_results['metric%d' % K][2], K, best_results['metric%d' % K][0],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]) + '\n')
if __name__ == '__main__':
    main()
