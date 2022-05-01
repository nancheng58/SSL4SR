from .base import AbstractTrainer
from .utils import recalls_ndcgs_and_mrr_for_ks
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F


class SASTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        # self.bce_criterion = nn.BCELoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.l2_emb = args.l2_emb

    def add_extra_loggers(self):
        # with torch.no_grad():
        #     dataiter = iter(self.train_loader)
        #     seqs, labels = dataiter.next()
        #
        #     self.writer.add_graph(self.model, seqs)
        pass

    @classmethod
    def code(cls):
        return 'sas'

    def close_training(self):
        self.train_loader.close()

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seq, pos, neg = batch
        seq, pos, neg = np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits = self.model(seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.args.device), torch.zeros(neg_logits.shape, device=self.args.device)

        indices = np.where(pos != 0)

        # logits = torch.stack([pos_logits, neg_logits], dim=-1)
        # probs = F.softmax(logits, dim=-1)

        # loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        # loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
        # loss = self.bce_criterion(probs[:, :, 0], pos_labels)

        loss = self.bce_criterion(pos_logits[indices], pos_labels[indices]) + self.bce_criterion(neg_logits[indices], neg_labels[indices])

        for param in self.model.parameters():
            loss += self.l2_emb * torch.norm(param)

        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        logits = self.model.predict(seqs.type(torch.long), candidates.type(torch.long))

        metrics = recalls_ndcgs_and_mrr_for_ks(logits, labels, self.metric_ks)

        return metrics
