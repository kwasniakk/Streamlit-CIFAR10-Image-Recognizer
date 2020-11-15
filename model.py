import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
import os

class Model(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.lr = 0.001
        self.model = models.resnet18(pretrained = True)

        self.model.fc = nn.Linear(512, 10, bias=True)
        self.criterion = nn.CrossEntropyLoss()
        self.training_losses = []

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.training_losses.append(loss.item())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        return {"val_loss": loss}

    """def epoch_end(self):
        train_loss_mean = np.mean(self.training_losses)
        self.logger.experiment.add_scalar('training_loss', train_loss_mean, global_step=self.current_epoch)
        self.training_losses = [] """ 

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_loss": avg_loss}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.lr)

class CIFAR10DataModule(pl.LightningDataModule):
    
    def __init__(self):
        super().__init__()
        self.data_dir = os.getcwd()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
        ])

    def prepare_data(self):
        CIFAR10(self.data_dir, train = True, download = True)
        CIFAR10(self.data_dir, train = False, download = True)
    
    def setup(self, stage = None):
        self.cifar_train = CIFAR10(self.data_dir, train = True, transform = self.transform)
        self.cifar_val = CIFAR10(self.data_dir, train = False, transform = self.transform)
        self.dims = self.cifar_train[0][0].shape

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size = 32)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_val, batch_size = 32)



class SaveActivations():
    activations = None
    def __init__(self, last_layer):
        self.hook = last_layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.data.numpy()
    def remove(self):
        self.hook.remove()
