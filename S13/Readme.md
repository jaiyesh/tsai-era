# Object Detection using YOLO V3

## Introduction 

We are using the dataset hosted within the GitHub repository named "Machine-Learning-Collection" by the user aladdinpersson. Specifically, the dataset is related to object detection using the YOLOv3 (You Only Look Once version 3) model, implemented using the PyTorch framework.

## Objective 

The objective of the project is to make several enhancements to an existing YOLOv3-based object detection system and deploy it using PyTorch Lightning and HuggingFace Spaces. 

## Hardware Used: Google Colab 

## [Hugging Face Space App Link: ]([https://huggingface.co/spaces/adil22jaleel/Yolo-V3-Pytorch-Lightning](https://huggingface.co/spaces/jaiyeshchahar/Yolo-Pytorch-Lighting))

## Code Structure

1. config.py

Configuration Parameters:

- DATASET: Specifies the dataset being used (PASCAL_VOC in this case).
- DEVICE: Sets the computing device to either "cuda" if available, otherwise "cpu".
- NUM_WORKERS: Number of CPU workers for data loading.
- BATCH_SIZE: Batch size for training.
- IMAGE_SIZE: The size to which input images will be resized.
- NUM_CLASSES: Number of classes in the dataset.
- LEARNING_RATE: Learning rate for the optimizer.
- WEIGHT_DECAY: Weight decay for regularization.
- NUM_EPOCHS: Number of training epochs.
- CONF_THRESHOLD: Confidence threshold for object detection.
- MAP_IOU_THRESH: IOU threshold for calculating Mean Average Precision (mAP).
- NMS_IOU_THRESH: IOU threshold for non-maximum suppression.
- S: List of anchor box scales for different YOLO layers.
- PIN_MEMORY: Whether to use pinned memory for data loading.
- LOAD_MODEL: Whether to load a pre-trained model.
- SAVE_MODEL: Whether to save checkpoints during training.
- CHECKPOINT_FILE: File name for checkpoint saving/loading.
- IMG_DIR: Directory containing images.
- LABEL_DIR: Directory containing labels.
- CHECKPOINT_PATH: Directory for saving checkpoints.
- ANCHORS: List of anchor boxes for each YOLO layer. Anchor boxes are specified as normalized coordinates within [0, 1].
- PASCAL_CLASSES: List of class labels corresponding to the PASCAL VOC dataset.

This configuration file sets up the necessary parameters and transformations to preprocess images and labels for training and testing an object detection model using YOLOv3. The provided anchor boxes and class labels are specific to the PASCAL VOC dataset, which contains various common object classes.

2. utils.py
The utils.py file contains a collection of utility functions that are used in conjunction with the YOLOv3 object detection model. 

- iou_width_height(boxes1, boxes2): This function calculates the Intersection over Union (IoU) of two sets of bounding boxes given their widths and heights.

- intersection_over_union(boxes_preds, boxes_labels, box_format): This function computes the IoU between predicted and labeled bounding boxes, considering whether the boxes are represented as midpoints or corners.

- non_max_suppression(bboxes, iou_threshold, threshold, box_format): Performs non-maximum suppression on a list of bounding boxes, removing redundant detections based on IoU and detection confidence.

- mean_average_precision(pred_boxes, true_boxes, iou_threshold, box_format, num_classes): Calculates the mean average precision (mAP) metric for object detection predictions.

- plot_image(image, boxes): Plots an image along with the predicted bounding boxes for visualization.

- get_evaluation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format, device): Computes predicted and true bounding boxes for evaluation using the given data loader and model.

- check_class_accuracy(model, loader, threshold): Calculates the accuracy of predicted classes, objectness, and non-objectness.

- get_mean_std(loader): Computes the mean and standard deviation of pixel values in the dataset.

- save_checkpoint(model, optimizer, filename): Saves model weights and optimizer state as a checkpoint.

- load_checkpoint(checkpoint_file, model, optimizer, lr): Loads model weights and optimizer state from a checkpoint file.

- get_loaders(train_csv_path, test_csv_path): Creates data loaders for training and testing datasets.

- plot_couple_examples(model, loader, thresh, iou_thresh, anchors): Plots examples from the dataset with predicted bounding boxes.

- xywhn2xyxy(x, w, h, padw, padh): Converts normalized bounding box coordinates to absolute coordinates.

- xyn2xy(x, w, h, padw, padh): Converts normalized segment coordinates to absolute coordinates.

- xyxy2xywhn(x, w, h, clip, eps): Converts bounding box coordinates from absolute to normalized.

- clip_boxes(boxes, shape): Clips bounding boxes to fit within specified image dimensions.

These utility functions provide essential functionalities for working with YOLOv3 models and performing various tasks related to object detection and evaluation.

3. dataset.py

The provided code defines two PyTorch datasets, YOLOTrainDataset and YOLOTestDataset, for loading images and their corresponding annotations from the Pascal VOC and MS COCO datasets.

4. loss.py

YoloLoss Class:

Initializes various loss functions and constants that weigh different components of the loss function differently (lambda_class, lambda_noobj, lambda_obj, lambda_box).


The loss computation consists of four main components: no-object loss, object loss, box coordinates loss, and class loss.


5. callback.py

The provided code defines three custom callback classes that can be used with PyTorch Lightning's training loop. Callbacks are functions or methods that are executed at specific points during training and can be used for various purposes such as logging, visualization, and custom metrics calculation. This is something very new that is learnt with the lightning module.

Main 3 callbaks are
- plot_examples_callback
- class_accuracy_callback
- map_callback

These custom callbacks provide additional functionality beyond the standard PyTorch Lightning callbacks, allowing you to perform specialized actions during training epochs. 

6. model.py

The provided code defines the implementation of the YOLOv3 architecture for object detection. 

- CNNBlock class:

This class defines a convolutional block with optional batch normalization and Leaky ReLU activation.
It is used to build the convolutional layers within the YOLOv3 architecture.

- ResidualBlock class:

This class defines a residual block with a series of convolutional layers.
The block can be set to use residual connections or not.
It's used to create the residual blocks within the architecture.

- ScalePrediction class:

This class defines the scale prediction layer that predicts bounding box coordinates and class probabilities.
The layer is used to generate predictions for object detection at different scales.

- YOLOv3 class:

This class defines the overall YOLOv3 architecture.
It creates the entire architecture based on the provided configuration.
It also handles the forward pass through the network, producing predictions at different scales.
The architecture incorporates residual blocks, upsampling, and scale prediction layers as specified in the configuration.
This class also includes an example of model instantiation and forward pass with random input data to verify the shapes of the output predictions.

7. yolo_lightning.py

Initialization and Model Setup:

The constructor initializes the Lightning module and sets up the YOLOv3 model and loss function.

- configure_optimizers:

This method configures the optimizer and learning rate scheduler for training.
It uses the Adam optimizer with weight decay and sets up a OneCycleLR scheduler.

- Data Loaders:

train_dataloader, val_dataloader, and test_dataloader methods define data loaders for training, validation, and testing.
These loaders are created using the get_loaders function from the utils module.

- Training Loop:

training_step method defines the training step.
Computes the loss for each scale and logs it for visualization.
Returns the loss for gradient optimization.

- Validation Loop:

validation_step method defines the validation step.
Computes the loss for each scale and logs it for visualization.

- Test Loop:

test_step method defines the test step.
Computes the loss for each scale and logs it for visualization.

- Checkpoint Loading:

on_train_start method loads a checkpoint if specified in the configuration.
It also computes scaled anchors based on the configuration parameters.
The YOLOv3Lightning module follows the PyTorch Lightning structure, making it easy to train and evaluate the YOLOv3 model while benefiting from Lightning's training loop abstractions and flexibility.

## Model Outcomes


### Train Metrics
- Loss: 3.670590877532959
- Class Accuracy: 85.670000%
- No Object Accuracy: 98.120000%
- Object Accuracy: 78.880000%

### Test Metrics

- Class Accuracy: 87.180000%
- No Object Accuracy: 98.230000%
- Object Accuracy: 75.690000%

### MAP Value

MAP (Mean Average Precision) is a widely used metric in object detection tasks, including YOLO, to evaluate the accuracy and quality of a detection model's predictions.

- MAP:  0.4017103314399719



