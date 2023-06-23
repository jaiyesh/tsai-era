# Session 8 Assignment

## Table of Contents
- [Problem Statement](#problem-statement)
- [Data Augmentation](#data-augmentation)
- [src](#src)
- [Notebook and Results](#notebook-and-results)
- [Findings](#findings)
- [Training curves and Misclassified Images](#training-images-and-misclassified-images)

## 🎯 Problem Statement
Assignment is:  
  
1. Change the dataset to CIFAR10  
2. Make this network:  
    1. C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10  
    2. Keep the parameter count less than 50000  
    3. Try and add one layer to another  
    4. Max Epochs is 20  
3. You are making 3 versions of the above code (in each case achieve above 70% accuracy):  
    1. Network with Group Normalization  
    2. Network with Layer Normalization  
    3. Network with Batch Normalization  
4. Share these details  
    1. Training accuracy for 3 models  
    2. Test accuracy for 3 models  
    3. Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images.  
5. write an explanatory README file that explains:
    1. what is your code all about,
    2. your findings for normalization techniques,
    3. add all your graphs
    4. your collection-of-misclassified-images 
6. Upload your complete assignment on GitHub and share the link on LMS


## 🎲 Data Augmentation 
Augmentation is performed using the transforms from pytorch . Three techniques are applied in the training data loader: Color Jitter, Random Horizontal, and Random Rotation.

## src
src contains model.py that have base model
and
utils.py that have train, test, and functions for plotting graphs and images

## 📓 Notebook and Results

Below is a table summarizing the performance of each model. It provides links to the notebooks for each model, along with their respective training and test accuracies.

| Model | Notebook Link | Training Accuracy | Test Accuracy |
|-------|---------------|-------------------|---------------|
| Batch Normalization | [BN Notebook](./Notebooks/Assignment8_Batch_Normalization.ipynb) | 76.30 % | 77.82 % |
| Group Normalization | [GN Notebook](./Notebooks/Assignment8_Gropu_Normalization.ipynb) | 75.20 % | 76.07 % |
| Layer Normalization | [LN Notebook](./Notebooks/Assignment8_layer_Normalizationh.ipynb) | 74.94 % | 76.92 % |


## 📈 Findings
Batch Normalization >Layer>Group.

## 📸 Training curves and Misclassified Images

### Batch Norm
training curves:
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/48a2d71a-33de-40f4-80f7-4d9681a886f9)

misclassified images:

![image](https://github.com/jaiyesh/tsai-era/assets/64524945/b0b78483-e51b-43da-8af7-6405f2622614)


### Group Norm

training curves:

![image](https://github.com/jaiyesh/tsai-era/assets/64524945/1b1e0c0b-e095-4972-98e7-bc756e2d6cb6)


misclassified images:

![image](https://github.com/jaiyesh/tsai-era/assets/64524945/582ad04b-27dc-4f2b-8e36-4387b37d12e3)


### Layer Norm

training curves:

![image](https://github.com/jaiyesh/tsai-era/assets/64524945/a535f90a-cb5e-41f2-9a38-a6228749fbde)



misclassified images:

![image](https://github.com/jaiyesh/tsai-era/assets/64524945/092c3423-fe85-43a9-b80e-ef13aa8d6bfc)

