import pytorch_lightning as pl
import torch
from util.options import parse_custom_options
from models.test_model import uorfTestGanModel, KmeansuorfTestGanModel, D1uorfTestGanModel, D2uorfTestGanModel, D3uorfTestGanModel, D4uorfTestGanModel, D5uorfTestGanModel, D6uorfTestGanModel, S1D2uorfTestGanModel, S1D1uorfTestGanModel, S2D1uorfTestGanModel, S2D2uorfTestGanModel

from data import MultiscenesDataModule

if __name__=='__main__':
    print('Parsing options...')
    opt = parse_custom_options()

    pl.seed_everything(opt.seed)

    print('Creating dataset...')
    dataset = MultiscenesDataModule(opt)  # create a dataset given opt.dataset_mode and other options

    print('Creating module...')
    # module = uorfTestGanModel(opt)
    training_version = opt.version
    if training_version == 'original':
        module = uorfTestGanModel(opt)
    elif training_version == 'kmeans':
        module = KmeansuorfTestGanModel(opt)
    elif training_version == 'decv1':
        module = D1uorfTestGanModel(opt)
    elif training_version == 'decv2':
        module = D2uorfTestGanModel(opt)
    elif training_version == 'decv3':
        module = D3uorfTestGanModel(opt)
    elif training_version == 'decv4':
        module = D4uorfTestGanModel(opt)
    elif training_version == 'decv5':
        module = D5uorfTestGanModel(opt)
    elif training_version == 'decv6':
        module = D6uorfTestGanModel(opt)
    elif training_version == 'ShapeAppV1DV1':
        module = S1D1uorfTestGanModel(opt)
    elif training_version == 'ShapeAppV2DV1':
        module = S2D1uorfTestGanModel(opt)
    elif training_version == 'ShapeAppV1DV2':
        module = S1D2uorfTestGanModel(opt)
    elif training_version == 'ShapeAppV2DV2':
        module = S2D2uorfTestGanModel(opt)
    else:
        print("Unknown training version: Shell script needs correct --version")

    ckpt = torch.load(opt.checkpoint)
    module.load_state_dict(ckpt["state_dict"], strict=False)

    trainer = pl.Trainer(
        gpus=opt.gpus,
        max_epochs=1)

    print('Start testing...')
    trainer.test(module, dataset)