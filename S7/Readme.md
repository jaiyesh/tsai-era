# ERA1 Session 7 Assignment

## Problem Statement

1. Your new target is:  
        1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)  
        2. Less than or equal to 15 Epochs  
        3. Less than 8000 Parameters  
        4. Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.  
2. Do this in exactly 3 steps  
3. Each File must have a "target, result, analysis" TEXT block (either at the start or the end)  
4. You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct.  
5. Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 
6. Explain your 3 steps using these targets, results, and analysis with links to your GitHub files (Colab files moved to GitHub).  
7. Keep Receptive field calculations handy for each of your models.  
8. If your GitHub folder structure or file_names are messy, -100.  
9. When ready, attempt SESSION 7 -Assignment Solution  


## Solution, Step 1 [Notebook](./ERA1_S7_step1.ipynb)

### Target   
- Set up the necessary components, including the dataset, data loader, training/testing steps, and log plots.
- Define a simple model with a Convolution block, Global Average Pooling (GAP), dropout, and batch normalization.

### Results
- Number of Parameters: 6,038
- Best Train Accuracy: 98.84%
- Best Test Accuracy: 99.01%

  ![image](https://github.com/jaiyesh/tsai-era/assets/64524945/4cf4ea30-2cbd-4b75-a00f-e9d976fb4e46)


### Analysis
- The model with 6,038 parameters achieved an accuracy of 99.01% in 15 epochs.
- The training and test accuracies are close, indicating that the model is not overfitting.

## Solution, Step 2 [Notebook](./ERA1_S7_step2.ipynb)

### Target   
- Incorporate image augmentation techniques such as random rotation and random affine to enhance the model's performance.

### Results
- Number of Parameters: 6,038
- Best Train Accuracy: 98.33%
- Best Test Accuracy: 99.19%

### Analysis
- Despite the inclusion of image augmentation techniques, the model with 6,038 parameters achieved a slightly improved accuracy of 99.19% in 15 epochs.
- The limited improvement could be due to the presence of dropout, which already serves a similar purpose.

## Solution, Step 3 [Notebook](./ERA1_S7_step3.ipynb)

### Target   
- Investigate the impact of incorporating the StepLR rate scheduler.
- Increase the model capacity by adding more convolutional layers.
- Optimize the learning rate and dropout value.

### Results
- Number of Parameters: 7,416
- Best Train Accuracy: 99.06%
- Best Test Accuracy: 99.33%

  ![image](https://github.com/jaiyesh/tsai-era/assets/64524945/02d963ad-a42f-4145-a5f9-97663f6bd098)


### Analysis
- By increasing the model capacity, optimizing the learning rate, and adding the StepLR rate scheduler, the model with 7,416 parameters achieved a cross-validated accuracy of 99.36% in 15 epochs.
- This model satisfies all the requirements regarding model size, accuracy, and number of epochs.
- Increasing the model capacity and incorporating the LR rate scheduler were crucial in achieving the desired accuracy within the specified timeframe.
