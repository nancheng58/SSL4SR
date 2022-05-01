import argparse

from options import args, parser
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import json
from os import path


def train():
    export_root = setup_train(args)

    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)

    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.train()
    trainer.test()


"""    
    #trainer.train()
    # test_result = test_with(trainer.best_model, test_loader)
    # save_test_result(export_root, test_result)
    trainer.test()
"""


def test():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)

    load_pretrained_weights(model, args.test_model_path, args.device)

    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.test()


if __name__ == '__main__':
    with open(path.normpath(args.config_file), 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        pass
