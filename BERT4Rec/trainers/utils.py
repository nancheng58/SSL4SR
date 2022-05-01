import torch

from utils import AverageMeterSet


def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


# B x C, B x C
def recalls_ndcgs_and_mrr_for_ks_length(scores, labels, ks, seq, length_lower_bound, length_upper_bound):
    metrics = {}
    scores = scores.cpu()
    labels = labels.cpu()
    filter_scores = []
    filter_labels = []
    scores = scores.tolist()
    labels = labels.tolist()
    seq = seq.tolist()
    for i in range(len(seq)):  # length filter
        length_seq = seq[i]
        if length_lower_bound <= length_seq < length_upper_bound:
            filter_scores.append(scores[i])
            filter_labels.append(labels[i])
    if len(filter_labels) == 0:
        for k in sorted(ks):
            metrics['Recall@%d' % k] = 0
            metrics['NDCG@%d' % k] = 0
            metrics['MRR@%d' % k] = 0
        return metrics
    scores = torch.tensor(data=filter_scores, device='cpu')
    labels = torch.tensor(data=filter_labels, device='cpu')
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
        ndcg = (dcg / idcg).mean().item()
        metrics['NDCG@%d' % k] = ndcg

        position_mrr = torch.arange(1, k + 1)
        weights_mrr = 1 / position_mrr.float()
        mrr = (hits * weights_mrr).sum(1)
        mrr = mrr.mean().item()

        metrics['MRR@%d' % k] = mrr
    # average_meter_set = AverageMeterSet()
    # for k, v in metrics.items():
    #     average_meter_set.update(k, v)
    # average_meter_set = average_meter_set.averages()
    # print(str(length_lower_bound))
    # print(average_meter_set)
    return metrics

# B x C, B x C
def recalls_ndcgs_and_mrr_for_ks(scores, labels, ks, seq):

    metrics = {}

    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
        ndcg = (dcg / idcg).mean().item()
        metrics['NDCG@%d' % k] = ndcg

        position_mrr = torch.arange(1, k + 1)
        weights_mrr = 1 / position_mrr.float()
        mrr = (hits * weights_mrr).sum(1)
        mrr = mrr.mean().item()

        metrics['MRR@%d' % k] = mrr

    return metrics
