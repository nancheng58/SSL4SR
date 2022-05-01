from .bert import BERTModel
from .sas import SASModel

MODELS = {
    BERTModel.code(): BERTModel,
    SASModel.code(): SASModel
}


def model_factory(args):
    model = MODELS[args.model_code]
    return model(args)
