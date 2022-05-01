# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""
import torch

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct, seq_len):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        dataobject = dataobject.get('rec.topk')
        length_lower_bound = [0, 20, 30, 40]
        length_upper_bound = [20, 30, 40, 51]
        for j in range(len(length_lower_bound)):
            print(j)
            metrics = {}
            filter_dataobject = []
            for i in range(len(seq_len)):  # length filter
                length_seq = seq_len[i+1]   # start at 1
                if length_lower_bound[j] <= length_seq < length_upper_bound[j]:
                    filter_dataobject.append(dataobject[i])  # start at 0
            filter_dataobject = torch.stack(filter_dataobject)
            for metric in self.metrics:
                metric_val = self.metric_class[metric].calculate_metric(filter_dataobject)
                print(metric_val)


        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict
