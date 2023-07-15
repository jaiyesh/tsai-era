import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from dataset import train_mean, train_std
from torchvision.transforms import Normalize

def get_incorrect_preds(model, test_dataloader):
  incorrect_examples = []
  pred_wrong = []
  true_wrong = []

  model.eval()
  for data,target in test_dataloader:
    data , target = data.cuda(), target.cuda()
    output = model(data)
    _, preds = torch.max(output,1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()
    preds = np.reshape(preds,(len(preds),1))
    target = np.reshape(target,(len(preds),1))
    data = data.cpu().numpy()
    for i in range(len(preds)):
        if(preds[i]!=target[i]):
            pred_wrong.append(preds[i])
            true_wrong.append(target[i])
            incorrect_examples.append(data[i])

  return true_wrong, incorrect_examples, pred_wrong

def plot_incorrect_preds(true,ima,pred,n_figures = 10):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
    
    denorm = Normalize((-train_mean / train_std).tolist(), (1.0 / train_std).tolist())
    print('Classes in order Actual and Predicted')
    n_row = int(n_figures/5)
    fig,axes = plt.subplots(figsize=(12, 3), nrows = n_row, ncols=5)
    plt.subplots_adjust(hspace=1)
    for ax in axes.flatten():
        a = random.randint(0,len(true)-1)
        image,correct,wrong = ima[a],true[a],pred[a]
        image = torch.from_numpy(image)
        image = denorm(image)*255
        image = image.permute(2, 1, 0) # from NHWC to NCHW
        correct = int(correct)
        wrong = int(wrong)
        image = image.squeeze().numpy().astype(np.uint8)
        im = ax.imshow(image) #, interpolation='nearest')
        ax.set_title(f'A: {labels[correct]} , P: {labels[wrong]}', fontsize = 8)
        ax.axis('off')
    plt.show()
    
def plot_sample_imgs(train_loader,n_figures = 40):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
    ima, targets = next(iter(train_loader))
    denorm = Normalize((-train_mean / train_std).tolist(), (1.0 / train_std).tolist())
    n_row = int(n_figures/10)
    fig,axes = plt.subplots(figsize=(10, 3), nrows = n_row, ncols=10)
    plt.subplots_adjust(hspace=1)
    for ax in axes.flatten():
        a = random.randint(0,len(ima)-1)
        image, target = ima[a], targets[a]
#         image = torch.from_numpy(image)
        image = denorm(image)*255
        image = image.permute(2, 1, 0) # from NHWC to NCHW
        image = image.squeeze().numpy().astype(np.uint8)
        im = ax.imshow(image) #, interpolation='nearest')
        ax.set_title(f'{labels[target]}', fontsize = 8)
        ax.axis('off')
    plt.show()