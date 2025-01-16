import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from transformers import RobertaForSequenceClassification, AdamW
from torchmetrics.classification import Accuracy

class SweetRoberta(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.config.model["pretrained-model"],
            torchscript = True,
            num_labels = self.config.model["num_labels"]
        )
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        logits = outputs.logits
        
        self.train_accuracy(logits, labels)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        logits = outputs.logits

        self.val_accuracy(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        val_acc = self.val_accuracy.compute()
        self.log("val_accuracy", val_acc, prog_bar=True)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.optimizer["lr"])
        return optimizer 
