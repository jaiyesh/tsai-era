# Imports
from torchvision import datasets
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


train_data = datasets.CIFAR10('./data', download=True, train=True)


x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
print(x.shape)
# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x, axis=(0, 1))/255
train_std = np.std(x, axis=(0, 1))/255
# print the mean and std
print(train_mean, train_std)

class Cifar10Dataset(datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, 
                    min_width=1, fill_value=train_mean, mask_fill_value = None),
    A.Normalize(mean=train_mean, std=train_std),
    ToTensorV2(),
])
    
test_transforms = A.Compose([
    A.Normalize(mean=train_mean, std=train_std),
    ToTensorV2(),
])

train_data = Cifar10Dataset('./data', download=True, train=True, transform=train_transforms)
test_data = Cifar10Dataset('./data', download=True, train=False, transform=test_transforms)

SEED = 1

cuda = torch.cuda.is_available()
print('cuda available?', cuda)

torch.manual_seed(SEED)

dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64) 


train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)