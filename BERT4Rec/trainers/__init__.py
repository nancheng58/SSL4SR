from .bert import BERTTrainer
from .sas import SASTrainer

TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    SASTrainer.code(): SASTrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.model_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
