import pytorch_lightning as pl
from model import Model, CIFAR10DataModule
from pytorch_lightning.loggers import TensorBoardLogger
import os

logger = TensorBoardLogger(
    save_dir=os.getcwd(),
    version=1,
    name='lightning_logs'
)


dataModule = CIFAR10DataModule()
model = Model()
for name, param in model.model.named_parameters():
    if ("bn" not in name):
        param.requires_grad = False
trainer = pl.Trainer(max_epochs = 15, gpus = 1, auto_lr_find = True, logger = logger)
trainer.tune(model, datamodule=dataModule)
trainer.fit(model, dataModule)
trainer.save_checkpoint("resnet.ckpt")

