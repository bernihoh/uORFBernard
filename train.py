import pytorch_lightning as pl
from util.options import parse_custom_options

from pytorch_lightning.callbacks import LearningRateMonitor

from models.train_model import KmeansuorfGanModel, ShapeAppV1DV1UorfGanModel, ShapeAppV2DV1UorfGanModel, ShapeAppV1DV2UorfGanModel, ShapeAppV2DV2UorfGanModel, Decv1uorfGanModel, Decv2uorfGanModel, Decv3uorfGanModel, Decv4uorfGanModel, Decv5uorfGanModel, Decv6uorfGanModel, uorfGanModel

from data import MultiscenesDataModule

if __name__=='__main__':
    print('Parsing options...')
    opt = parse_custom_options()

    pl.seed_everything(opt.seed)

    print('Creating dataset...')
    dataset = MultiscenesDataModule(opt)  # create a dataset given opt.dataset_mode and other options

    print('Creating module...')
    training_version = opt.version
    if training_version == 'original':
        module = uorfGanModel(opt)
    elif training_version == 'kmeans':
        module = KmeansuorfGanModel(opt)
    elif training_version == 'decv1':
        module = Decv1uorfGanModel(opt)
    elif training_version == 'decv2':
        module = Decv2uorfGanModel(opt)
    elif training_version == 'decv3':
        module = Decv3uorfGanModel(opt)
    elif training_version == 'decv4':
        module = Decv4uorfGanModel(opt)
    elif training_version == 'decv5':
        module = Decv5uorfGanModel(opt)
    elif training_version == 'decv6':
        module = Decv6uorfGanModel(opt)
    elif training_version == 'ShapeAppV1DV1':
        module = ShapeAppV1DV1UorfGanModel(opt)
    elif training_version == 'ShapeAppV2DV1':
        module = ShapeAppV2DV1UorfGanModel(opt)
    elif training_version == 'ShapeAppV1DV2':
        module = ShapeAppV1DV2UorfGanModel(opt)
    elif training_version == 'ShapeAppV2DV2':
        module = ShapeAppV2DV2UorfGanModel(opt)
    else:
        print("Unknown training version: Shell script needs correct --version")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus=opt.gpus,
        strategy="ddp", # ddp_spawn
        max_epochs=opt.niter + opt.niter_decay + 1,
        callbacks=[lr_monitor])

    print('Start training...')
    trainer.fit(module, dataset)
