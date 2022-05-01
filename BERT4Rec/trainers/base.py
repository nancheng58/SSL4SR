import json

from loggers import *
from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from trainers.utils import recalls_ndcgs_and_mrr_for_ks_length
from utils import AverageMeterSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from abc import *
from pathlib import Path


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.optimizer = self._create_optimizer()

        if args.resume_path is not None:
            print("resuming model and optimizer\'s parameters")

            self.checkpoint = torch.load(Path(args.resume_path), map_location=torch.device(args.device))
            print("checkpoint epoch number: ", self.checkpoint['epoch'])
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict((self.checkpoint['optimizer_state_dict']))

        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric

        self.export_root = export_root
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.add_extra_loggers()

        self.logger_service = LoggerService(self.train_loggers, self.val_loggers)
        self.log_period_as_iter = args.log_period_as_iter
        self.batch_size = args.train_batch_size

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    @abstractmethod
    def calculate_metrics_length(self, batch):
        pass

    @abstractmethod
    def close_training(self):
        pass

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            print("epoch: ", epoch)
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)

            self.lr_scheduler.step()

        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()
        self.close_training()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()

        average_meter_set = AverageMeterSet()
        # tqdm_dataloader = tqdm(self.train_loader)

        iterator = self.train_loader if not self.args.show_process_bar else tqdm(self.train_loader)

        tot_loss = 0.
        tot_batch = 0

        for batch_idx, batch in enumerate(iterator):
            # batch_size = batch[0].size(0)
            # batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)

            tot_loss += loss.item()

            tot_batch += 1

            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())

            if self.args.show_process_bar:
                iterator.set_description('Epoch {}, loss {:.3f} '.format(epoch + 1, average_meter_set['loss'].avg))

            accum_iter += self.batch_size

            self.writer.add_scalar("learning_rate", self.get_lr(), accum_iter)

            if self._needs_to_log(accum_iter):
                if self.args.show_process_bar:
                    iterator.set_description('Logging to Tensorboard')

                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        print(tot_loss)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            iterator = self.val_loader if not self.args.show_process_bar else tqdm(self.val_loader)

            for batch_idx, batch in enumerate(iterator):

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

                if self.args.show_process_bar:
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                          ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                          ['MRR@%d' % k for k in self.metric_ks[:3]]
                    description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace('Recall', 'R').replace('MRR', 'M')
                    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                    iterator.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            self.logger_service.log_val(log_data)
            average_metrics = average_meter_set.averages()
            print(average_metrics)

    def test(self):
        print('Test best model with test set!')

        if self.args.test_model_path is not None:
            best_model = torch.load(Path(self.args.test_model_path), map_location=torch.device(self.args.device)).get(
                'model_state_dict')
        else:
            best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get(
                'model_state_dict')

        self.model.load_state_dict(best_model)

        self.model.eval()

        average_meter_set = AverageMeterSet()
        average_meter_set0 = AverageMeterSet()
        average_meter_set1 = AverageMeterSet()
        average_meter_set2 = AverageMeterSet()
        average_meter_set3 = AverageMeterSet()
        average_meter_set_list = [average_meter_set0, average_meter_set1, average_meter_set2, average_meter_set3]
        with torch.no_grad():
            iterator = self.test_loader if not self.args.show_process_bar else tqdm(self.test_loader)

            for batch_idx, batch in enumerate(iterator):

                metrics = self.calculate_metrics(batch)
                metrics_length = self.calculate_metrics_length(batch)
                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                index = 0
                for metri in metrics_length:
                    for k, v in metri.items():
                        average_meter_set_list[index].update(k, v)
                    index += 1
                if self.args.show_process_bar:
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] + \
                                          ['Recall@%d' % k for k in self.metric_ks[:3]] + \
                                          ['MRR@%d' % k for k in self.metric_ks[:3]]
                    description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace('Recall', 'R').replace('MRR', 'M')
                    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                    iterator.set_description(description)
            for index in range(len(average_meter_set_list)):
                average_metrics_length = average_meter_set_list[index].averages()
                print(str(index))
                print(average_metrics_length)
            average_metrics = average_meter_set.averages()
            print(average_metrics)
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f)

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                             momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs', 'tensorboard_visualization'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss', graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
