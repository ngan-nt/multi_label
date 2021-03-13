import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# import custom architectures
from src.architectures.xlmr_encoder import XLMREncoder
from pytorch_lightning.metrics.classification import Accuracy, F1
from torch.nn import BCEWithLogitsLoss, BCELoss

class LitModelIntent(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.loss = BCELoss(reduction='mean')
        self.accuracy = Accuracy()

        # Initialize model architecture
        self.architecture = XLMREncoder(hparams=self.hparams)
    
    def forward(self, x):
        return self.architecture(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        input_ids, _, _, y = batch
        logits = self.architecture(input_ids)
        loss = self.loss(logits, y)
        
        # training metrics
        acc = self.accuracy(logits, y.long())

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        input_ids, _, _, y = batch
        logits = self.architecture(input_ids)
        # print(logits)
        loss = self.loss(logits, y)

        # validation metrics
        acc = self.accuracy(logits, y.long())

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return {"batch_val_loss": loss, "batch_val_acc": acc, "batch_val_preds": logits, "batch_val_y": y}

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        input_ids, _, _, y = batch
        logits = self.architecture(input_ids)
        loss = self.loss(logits, y)

        # test metrics
        acc = self.accuracy(logits, y.long())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
 
