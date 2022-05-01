from .base import BaseModel
import torch.nn as nn
from .sas_model.sas import SAS


class SASModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.sas = SAS(args)

    @classmethod
    def code(cls):
        return 'sas'

    def forward(self, log_seqs, pos_seqs, neg_seqs):  # for training
        return self.sas(log_seqs, pos_seqs, neg_seqs)

    def predict(self, log_seqs, item_indices):  # for inference
        return self.sas.predict(log_seqs, item_indices)
