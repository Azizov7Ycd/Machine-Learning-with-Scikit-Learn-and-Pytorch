# Importing the necessary libraries
import numpy as np
import scipy.signal
import torch 
from torchvision.io import read_image
import torch.nn as nn
from torchvision import transforms
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Naive function for convution of two vectors
def conv1d(x,w,p=0,s=1):
    w_rot=np.array(w[::-1])
    x_array=np.array(x)
    if p>0:
        zero_pad=np.zeros(shape=p)
        x_padded=np.concatenate([zero_pad,x_array,zero_pad])
    else:
        x_padded=x_array
    res=[]
    for i in range(0, int((len(x_padded)-len(w_rot)))+1,s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot))
    return np.array(res)
# Testing the convolution function: parameters padding=0, stride 1
x=[1,7,1,3,4,89,0,9,16,5]
w=[1,9,7]
convul=conv1d(x,w)
print('Conv1d implementation',conv1d(x,w))
# Implementing NumPy function 
print('NumPy results:',np.convolve(np.array(x),np.array(w),mode='same'))

# Defining a convolution function 
def con2d(X,W,p=(0,0),s=(1,1)):
    W_rot=np.array(W)[::-1,::-1]
    X_orig=np.array(X)
    n_1=X_orig.shape[0]+2*p[0]
    n_2=X_orig.shape[1]+2*p[1]
    X_padded=np.zeros(shape=(n_1,n_2))
    X_padded[p[0]:p[0]+X_orig.shape[0],p[1]:p[0]+X_orig.shape[0]]=X_orig
    res=[]
    for i in range(0,int((X_padded.shape[0] - W_rot.shape[0])/s[0])+1, s[0]):
        res.append([])
    for j in range(0,int((X_padded.shape[1] -W_rot.shape[1])/s[1])+1, s[1]):
        X_sub = X_padded[i:i+W_rot.shape[0],j:j+W_rot.shape[1]]
        res[-1].append(np.sum(X_sub * W_rot))
    return(np.array(res))
X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]
print('Conv2d implementation',con2d(X,W))
# Printing the SciPy results
print('SciPy results\n',scipy.signal.convolve2d(X,W,mode='same'))

# Reading an image using torchvision read_image function 
image=read_image(r"C:\Users\User\OneDrive\Dokumente\Bachelor Thesis\Writings for thesis\Last figure.png")
print("Image shape",image.shape)
print('Number of channels',image.shape[0])
print('Image data type',image.dtype)
print(image[:,100:102,100:102])

# Regularization in NN
loss_fn=nn.BCELoss()
loss=loss_fn(torch.tensor([0.9]),torch.tensor([0.1]))
l2_lambda=0.001
# Convolutional layer
conv_layer=nn.Conv2d(in_channels=3,out_channels=5,kernel_size=5)
l_2=l2_lambda*[(p**2).sum() for p in conv_layer.parameters()]
total_loss_CNN=loss+l_2
# Linear Layer
linear_layer=nn.Linear(16,10)
l_2=l2_lambda*[(p**2).sum() for p in linear_layer.parameters()]
total_loss_Linear=loss+l_2

# Binary Cross Entropy=> binary classification problem 
logits=torch.tensor([0.8])
probas=torch.sigmoid(logits)
target=torch.tensor([1.0])
bce_loss_fn=nn.BCELoss()
bce_logits_loss_fn=nn.BCEWithLogitsLoss()
print('BCE w Probas',bce_loss_fn(probas,target))
print('BCE w Logits',bce_logits_loss_fn(logits,target))
## Categotical Cross Entropy=> multiple classification problem
logits=torch.tensor([[1.5,0.8,2.1]])
probas=torch.softmax(logits,dim=1) # dimension needs to be specified
target=torch.tensor([2])
cce_fn_logits=nn.CrossEntropyLoss()
cce_fn_probas=nn.NLLLoss()
print(f'CCE w Logits {cce_fn_logits(logits,target):.4f}')
print(f'CCE w Probas {cce_fn_probas(torch.log(probas),target):.4f}') 

# Loading and preprocessing the data
image_path='./'
transform=transforms.Compose([transforms.ToTensor()])
# Loading train and test MNIST datsets
mnist_train_dataset=torchvision.datasets.MNIST(root=image_path,train=True,transform=transform,download=True)
mnist_test_dataset=torchvision.datasets.MNIST(root=image_path,train=False,transform=transform,download=True)
# Looking at one of the pictures from the train dataset
picture,label=mnist_train_dataset[0]
picture.shape
len(mnist_train_dataset) # 60000 MNIST images with labels
# Getting a validation subset from train dataset 
mnist_train_dataset,mnist_valid_dataset=train_test_split(mnist_train_dataset,test_size=10000,shuffle=True,random_state=42)
# Retrieving 10 labels from the validation dataset
[lab for (image,lab) in mnist_valid_dataset[:10]]
# DataLoaders with batches of 64 
batch_size=64
torch.manual_seed(1)
train_dl=DataLoader(mnist_train_dataset,batch_size=batch_size,shuffle=True)
valid_dl=DataLoader(mnist_valid_dataset,batch_size=batch_size,shuffle=True)
test_dl=DataLoader(mnist_test_dataset,batch_size=batch_size,shuffle=True)

# Creating a convolutional model 
# Firts convolutional layer=> 32 feature maps, second convolutional layer=> 64 feature maps
model=nn.Sequential()
model.add_module('conv1',nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5, padding=2))
model.add_module('relu1',nn.ReLU())
model.add_module('pool1',nn.MaxPool2d(kernel_size=2))
model.add_module('conv2',nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2))
model.add_module('relu2',nn.ReLU())
model.add_module('pool2',nn.MaxPool2d(kernel_size=2))
# Looking at the model output 
x=torch.ones((4,1,28,28))
model(x).shape #output shape:[4,64,7,7] (batch, feature maps, height, width)
# Falttening to pass the input to Linear layer
model.add_module('flatten',nn.Flatten())
# Model output after flattening
x=torch.ones((4,1,28,28))
model(x).shape  # output shape:[4,3136] (batch,input_units)
# Adding fully connected layers with dropout layers in between 
model.add_module('fc1',nn.Linear(3136,1024))
model.add_module('relu3',nn.ReLU())
model.add_module('dropout',nn.Dropout(p=0.5))
model.add_module('fc2',nn.Linear(1024,10))
# model.add_module('sigmoid',nn.Softmax())

# Creating a loss function and an optimizer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# Looking at the model layers
model[4]

# Defining a train function for the model 
def train(model,num_epochs,train_dl,valid_dl):
    loss_hist_train=[0]*num_epochs
    loss_hist_val=[0]*num_epochs
    acc_hist_train=[0]*num_epochs
    acc_hist_val=[0]*num_epochs
    for epoch in range(num_epochs):
        model.train() # Droping the neurons during training process
        for x_batch,y_batch in train_dl:
            pred=model(x_batch)
            loss=loss_fn(pred,y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch]+=loss.item()
            is_correct=(torch.argmax(pred,dim=1)==y_batch).float()
            acc_hist_train[epoch]+=is_correct.sum().item()
        # Dividing through the length of the train dataloader (number of batches) would return the average loss per sample
        loss_hist_train[epoch]/=len(train_dl)
        # Getting accuracy as a fraction after dividing through the whole dataset 
        acc_hist_train[epoch]/=len(train_dl.dataset)
        # Evaluating the model=> Dropout layer is inactive
        model.eval()
        with torch.no_grad():
            for x_batch,y_batch in valid_dl:
                pred1=model(x_batch)
                loss1=loss_fn(pred1,y_batch)
                loss_hist_val[epoch]+=loss1.item()
                is_correct=(torch.argmax(pred1,dim=1)==y_batch).float()
                acc_hist_val[epoch]+=is_correct.sum().item()
        # Dividing through the number of batches to determine loss per sample
        loss_hist_val[epoch]/=len(valid_dl)
        # Dividing through the number of samples=> to detemine accuracy as a float
        acc_hist_val[epoch]/=len(valid_dl.dataset)
        # Print accuracy and loss for every 20 epochs
        print(f'{epoch}, Training Loss: {loss_hist_train[epoch]:.3f}, Training Accuracy {acc_hist_train[epoch]:.3f},\n Valid Loss: {loss_hist_val[epoch]:.3f}, Test Accuracy: {acc_hist_val[epoch]:.3f}')
    return loss_hist_train,loss_hist_val,acc_hist_train,acc_hist_val

# Training our model 
torch.manual_seed(44)
num_epochs=10
hist=train(model,num_epochs=num_epochs,train_dl=train_dl,valid_dl=valid_dl)