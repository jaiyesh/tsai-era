import lightning.pytorch as pl
import config
from utils import (check_class_accuracy,get_evaluation_bboxes,mean_average_precision,plot_couple_examples)
from lightning.pytorch.callbacks import Callback


class plot_examples_callback(Callback):
    def __init__(self, epoch_interval: int = 5) -> None:
        super().__init__()
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            plot_couple_examples(
                model=pl_module,
                loader=pl_module.train_dataloader(),
                thresh=0.6,
                iou_thresh=0.5,
                anchors=pl_module.scaled_anchors,
            )


class class_accuracy_callback(pl.Callback):
    def __init__(self, train_epoch_interval: int = 1, test_epoch_interval: int = 10) -> None:
        super().__init__()
        self.train_epoch_interval = train_epoch_interval
        self.test_epoch_interval = test_epoch_interval

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.train_epoch_interval == 0:
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(model=pl_module, loader=pl_module.train_dataloader(), threshold=config.CONF_THRESHOLD)
            class_acc = round(class_acc.item(),2)
            no_obj_acc = round(no_obj_acc.item(),2)
            obj_acc = round(obj_acc.item(),2)

            pl_module.log_dict(
                {
                    "train_class_acc": class_acc,
                    "train_no_obj_acc": no_obj_acc,
                    "train_obj_acc": obj_acc,
                },
                logger=True,
            )
            print(f"Logged on A100 GPU's Colab- Adil Jaleel")
            print(f"Epoch Number: {trainer.current_epoch + 1}")
            print("Train Metrics")
            print(f"Loss: {trainer.callback_metrics['train_loss_epoch']}")
            print(f"Class Accuracy: {class_acc:2f}%")
            print(f"No Object Accuracy: {no_obj_acc:2f}%")
            print(f"Object Accuracy: {obj_acc:2f}%")

        if (trainer.current_epoch + 1) % self.test_epoch_interval == 0:
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(model=pl_module, loader=pl_module.test_dataloader(), threshold=config.CONF_THRESHOLD)
            class_acc = round(class_acc.item(),2)
            no_obj_acc = round(no_obj_acc.item(),2)
            obj_acc = round(obj_acc.item(),2)
            
            pl_module.log_dict(
                {
                    "test_class_acc": class_acc,
                    "test_no_obj_acc": no_obj_acc,
                    "test_obj_acc": obj_acc,
                },
                logger=True,
            )

            print("Test Metrics")
            print(f"Class Accuracy: {class_acc:2f}%")
            print(f"No Object Accuracy: {no_obj_acc:2f}%")
            print(f"Object Accuracy: {obj_acc:2f}%")

class map_callback(pl.Callback):
    def __init__(self, epoch_interval: int = 10) -> None:
        super().__init__()
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            pred_boxes, true_boxes = get_evaluation_bboxes(loader=pl_module.test_dataloader(), model=pl_module, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD, device=config.DEVICE,)

            map_val = mean_average_precision(pred_boxes=pred_boxes, true_boxes=true_boxes, iou_threshold=config.MAP_IOU_THRESH, box_format="midpoint", num_classes=config.NUM_CLASSES)
            print("MAP: ", map_val.item())
            pl_module.log("MAP",map_val.item(),logger=True)
            pl_module.train()