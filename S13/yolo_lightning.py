import torch
import torch.optim as optim
import lightning.pytorch as pl
from tqdm import tqdm
from model import YOLOv3
from loss import YoloLoss
from utils import get_loaders, load_checkpoint, check_class_accuracy, intersection_over_union
import config
from torch.optim.lr_scheduler import OneCycleLR


class YOLOv3Lightning(pl.LightningModule):
    def __init__(self, config, lr_value=0):
        super().__init__()
        self.automatic_optimization =True
        self.config = config
        self.model = YOLOv3(num_classes=self.config.NUM_CLASSES)
        self.loss_fn = YoloLoss()

        if lr_value == 0:
          self.learning_rate = self.config.LEARNING_RATE
        else:
          self.learning_rate = lr_value

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        EPOCHS = self.config.NUM_EPOCHS * 2 // 5
        scheduler = OneCycleLR(optimizer, max_lr=1E-3, steps_per_epoch=len(self.train_dataloader()), epochs=EPOCHS, pct_start=5/EPOCHS, div_factor=100, three_phase=False, final_div_factor=100, anneal_strategy='linear')
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def train_dataloader(self):
        train_loader, _, _ = get_loaders(
            train_csv_path=self.config.DATASET + "/train.csv",
            test_csv_path=self.config.DATASET + "/test.csv",
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (y[0].to(self.device),y[1].to(self.device),y[2].to(self.device))
        out = self(x)

        loss = (self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2]))
        
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def val_dataloader(self):
        _,  _, val_loader = get_loaders(
            train_csv_path=self.config.DATASET + "/train.csv",
            test_csv_path=self.config.DATASET + "/test.csv",
        )

        return val_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (
            y[0].to(self.device),
            y[1].to(self.device),
            y[2].to(self.device),
        )
        out = self(x)
        loss = (
            self.loss_fn(out[0], y0, self.scaled_anchors[0])
            + self.loss_fn(out[1], y1, self.scaled_anchors[1])
            + self.loss_fn(out[2], y2, self.scaled_anchors[2])
        )

        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
  

    def test_dataloader(self):
        _, test_loader, _ = get_loaders(
            train_csv_path=self.config.DATASET + "/train.csv",
            test_csv_path=self.config.DATASET + "/test.csv",
        )
        return test_loader

    def test_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (
            y[0].to(self.device),
            y[1].to(self.device),
            y[2].to(self.device),
        )
        out = self(x)
        loss = (
            self.loss_fn(out[0], y0, self.scaled_anchors[0])
            + self.loss_fn(out[1], y1, self.scaled_anchors[1])
            + self.loss_fn(out[2], y2, self.scaled_anchors[2])
        )
        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def on_train_start(self):
        if self.config.LOAD_MODEL:
            load_checkpoint(self.config.CHECKPOINT_FILE, self.model, self.optimizers(), self.config.LEARNING_RATE)
        self.scaled_anchors = (
            torch.tensor(self.config.ANCHORS)
            * torch.tensor(self.config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(self.device)