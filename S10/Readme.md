# ERA1_Session10
## Assignment
    1. Write a ResNet architecture for CIFAR10 that has the following architecture: 
        1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k] 
        2. Layer1 - 
            1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k] 
            2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
            3. Add(X, R1) 
        3. Layer 2 - 
            1. Conv 3x3 [256k] 
            2. MaxPooling2D 
            3. BN 
            4. ReLU 
        4. Layer 3 - 
            1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k] 
            2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k] 
            3. Add(X, R2) 
        5. MaxPooling with Kernel Size 4 
        6. FC Layer
        7. SoftMax 
    2. Uses One Cycle Policy such that: 
        1. Total Epochs = 24 
        2. Max at Epoch = 5 
        3. LRMIN = FIND 
        4. LRMAX = FIND 
        5. NO Annihilation 
    3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8) 
    4. Batch size = 512 
    5. Use ADAM, and CrossEntropyLoss 
    6. Target Accuracy: 90% 

## Solution - Notebook https://github.com/jaiyesh/tsai-era/blob/main/S10/s10_assignment.ipynb
In this we will use above given CNN architecture to achieve target test accuracy of more than 90% from the 19th epoch.

Following are the model parameter, train and test accuracies achieved in training the model for 24 epochs.
- Model Parameters - 6,573,130
- Train Accuracy - 98.60%
- Test Accuracy - 93.07%


### Graph showing LR finder to determine the optimum learning rate to use in one cycle LR policy.
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/98d8637c-d437-4c2d-8e8a-ba4288ca1e21)



### Graphs from training showing loss and accuracy for train and test datasets
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/8153764a-9725-4bf3-b9b7-249b931ac795)


### Graph showing one cycle learning-rate(LR) changes during training 
![image](https://github.com/jaiyesh/tsai-era/assets/64524945/7e812248-2365-4ff4-8bbe-c54c68ea4285)


### Ten misclassified images from the trained model.
![Misclassied images]
