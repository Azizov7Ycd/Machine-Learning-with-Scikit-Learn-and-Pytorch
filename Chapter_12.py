# Importing the necessary libraries
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
import os 
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
import torchvision 
from itertools import islice
from torch.utils.data import TensorDataset
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Changing the precision 
np.set_printoptions(precision=3)
a=[1,2,3]
b=np.array([1,2,3],dtype=np.int32)
# Creating tensors from the given objects
t_a=torch.tensor(a)
t_b=torch.from_numpy(b)
print(t_a)
print(t_b)
# .shape attribute allows us to retrieve tensor's shape
t_ones=torch.ones(2,3)
t_ones.shape # returns size [2,3]
print(t_ones) # printing the PyTorch tensor
# Creating a tensor with random numbers
ran_ten=torch.rand(4,4)
print(ran_ten)
# .to method to change the dtype to a desired one
new_ran=ran_ten.to(torch.int64)
print(new_ran.dtype)

# Transposing a PyTorch tensor
ten=torch.rand(3,5)
t_ten=torch.transpose(ten,0,1)
print(t_ten.shape)
# Reshaping a PyTorch tensor
ten0=torch.zeros(3,4)
ten_re=ten0.reshape(2,6)
print(ten_re)
# Squeezing tensor; removing dimensions with only 1 object 
ten=torch.rand(2,5,1,1,4)
ten_n=torch.squeeze(ten)
print(ten_n.shape)
print(ten_n.ndim) 

# Elementwise multiplication of the tensors
torch.manual_seed(1)
t1=2*torch.rand(3,4)+1
t2=torch.normal(mean=0,std=1,size=(3,4))
t_mul=torch.multiply(t1,t2)
print(t_mul)  # Printing the result
# Calculating mean, sum or std of a tensor
t1=torch.normal(size=(6,6),mean=0,std=1)
mean_v=torch.mean(t1,axis=0)
print(mean_v)  # mean values per column 
# Matrix multiplication of the tensors
t5=torch.matmul(t1,torch.transpose(t2,0,1))
print(t5)
t6=torch.matmul(torch.transpose(t1,0,1),t2)
print(t6)
# Calculating the matrix norm 
t_1=torch.rand(5,5)
t_norm=torch.linalg.norm(t_1,ord=2,dim=1)
print(t_norm)

# Splitting the tensor 
torch.manual_seed(1)
te1=torch.rand(4,5)
t_splits=torch.chunk(te1,chunks=4)
# iterating over the chunks
[item.numpy() for item in t_splits] 
# using torch.split function 
te2=torch.rand(7)
t_splits1=torch.split(te2,split_size_or_sections=[5,2])
[item.numpy() for item in t_splits1]
# Cancatenating the tensors
t_ones=torch.ones(4,4)
t_zeros=torch.zeros(4,4)
t_con=torch.cat((t_ones,t_zeros),axis=1)
print(t_con.numpy())
# Stacking the tensors
A=torch.ones(3)
B=torch.zeros(3)
res=torch.stack((A,B),dim=1)
print(res.shape)

# Dataloader enables iteration through the dataset 
t=torch.arange(6,dtype=torch.int32)
data_loader=DataLoader(t)
# iterating through the dataloader
for item in data_loader:
    print(item)
# Dataloader can be used for batching 
data_loader1=DataLoader(t,batch_size=3,drop_last=False)
for i,batch in enumerate(data_loader1):
    print(f'Batch {i} : {batch}')
# Combining tensors into joint dataset 
# Often combiniting feature values and class labels
t_x=torch.randn((4,3),dtype=torch.float32)
t_y=torch.arange(4)
# Joining the datasets
class Jointdataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    # underscores before and after the function allow differentiation between built-in and customized functions
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
# Creating a joint dataset with custom datasets as follows 
join_data=Jointdataset(t_x,t_y)
for i in range(join_data.__len__()):
    print(f'x : {join_data.__getitem__(i)[0]}; y : {join_data.__getitem__(i)[1]}')

# Shuffling the data
# Thus the rows are shuffled without losing the correspondence between x and y 
torch.manual_seed(1)
data_loader2=DataLoader(dataset=join_data,batch_size=2,shuffle=True)
for i,item in enumerate(data_loader2,1):
    print(f'tensor: {item[0]}, label: {item[1]}')
# It is often necesssary to iterate over the batches more than once
for epoch in range(4):
    print(f'Epoch {epoch+1}')
    for _,batch in enumerate(data_loader2):
        print(f'x : {batch[0]}, y : {batch[1]}')

# Creating a dataset from the local storage 
# Getting the current working directory 
imig_path=Path(r'.\cat_dog_pictures')
file_list=sorted([name for name in imig_path.glob('*.jpg')])
print(file_list)
# Plotting dog/cat images
fig,ax=plt.subplots(nrows=2,ncols=3,figsize=(10,5))
ax=ax.flatten()
for i,im in enumerate(file_list):
    ax[i].imshow(Image.open(im))
    print('Image shape :',np.array(Image.open(im)).shape)
    ax[i].set_title(im.stem,size=15)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
# Assigning the label 1 to dogs and 0 to cats
labels=[1 if 'dog' in p.stem else 0 for p in file_list]
print(labels)

# Joining the labels and the data with each other 
class ImageDataset(Dataset):
    def __init__(self,file_list,labels):
        self.file_list=file_list
        self.labels=labels
    def __getitem__(self, index):
        file=self.file_list[index]
        label=self.labels[index]
        return file,label
    def __len__(self):
        return len(self.labels)
# Creating acoupled dataset from labels and files
dat=ImageDataset(file_list,labels)
for file,label in dat:
    print(file,label)

# Loading image contents; resizing images to desired size 80*120
img_height,img_width=80,120
# Cropping the image; Transforming the image to tensor
transform=transforms.Compose([transforms.Resize(size=(img_height,img_width)),transforms.ToTensor()])
# Updating the ImageDataset class with the self defined transform 
class ImageDataset(Dataset):
    def __init__(self,file_list,labels,transform=None):
        self.labels=labels
        self.file_list=file_list
        self.transform=transform
    def __getitem__(self,index):
        img=Image.open(self.file_list[index])
        # Transforming the image if transformer determined
        if self.transform is not None:
            img=self.transform(img)
        label=self.labels[index]
        return img,label
    def __len__(self):
        return len(self.labels)
# Creating an image dataset 
image_dataset=ImageDataset(file_list,labels,transform)
# Vizualising the transformed images using Matplotlib 
fig,ax=plt.subplots(nrows=2,ncols=3,figsize=(10,5))
ax=ax.flatten()
for i,example in enumerate(image_dataset):
    ax[i].imshow(example[0].numpy().transpose((1,2,0)))
    ax[i].set_title(f'{example[1]}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()

# Storing the CelA dataset 
image_path=Path(r'./')
image_path.is_dir()
celeba_dataset=torchvision.datasets.CelebA(image_path,split='train',target_type='attr',download=False)
# Checking if object is of object is of torch.utils.data.Dataset
assert isinstance(celeba_dataset,torch.utils.data.Dataset)
# Looking how the training example looks like
example=next(iter(celeba_dataset)) 
print(example)
# Plotting 18 pictures as well their labels
fig,ax=plt.subplots(nrows=3,ncols=6,figsize=(12,4))
ax=ax.flatten()
for i,(image,attributes) in enumerate(islice(celeba_dataset,18)):
    ax[i].imshow(image)
    # Binary attribute corresponding to smiling is the 31 element of all
    ax[i].set_title(f'Smile {attributes[31]}',size=15)
    ax[i].set_xticks([]),ax[i].set_yticks([])
plt.tight_layout()
# Fetching MNIST dataset 
mnist_dataset=torchvision.datasets.MNIST(image_path,'train',download=True)
assert isinstance(mnist_dataset,torch.utils.data.Dataset)
example=next(iter(mnist_dataset))
print(example)
# Plotting 10 MNIST values
fig,ax=plt.subplots(nrows=2,ncols=5,figsize=(15,4))
ax=ax.flatten()
for i,(image,label) in enumerate(islice(mnist_dataset,10)):
    ax[i].imshow(image,cmap='gray_r')
    ax[i].set_title(label,size=15)
    ax[i].set_xlabel([]),ax[i].set_ylabel([])
plt.tight_layout()

# Building NN model in PyTorch 
# Creating a toy example for linear regression 
X=np.arange(10,dtype=np.float32).reshape(10,1)
Y=np.array([1.0, 1.3, 3.1, 2.0, 5.0,6.3, 6.6,7.4, 8.0,9.0], dtype=np.float32)
plt.scatter(X,Y,marker='o',color='blue',s=18)
plt.xlabel('x',size=15)
plt.ylabel('y',size=15)
plt.tight_layout()
# Standardizing the data
X_train_std=(X-X.mean())/X.std()
X_train_tens=torch.from_numpy(X_train_std)
y_tens=torch.from_numpy(Y)
train_ds=TensorDataset(X_train_tens,y_tens)
batch_size=1 
train_dl=DataLoader(train_ds,batch_size,shuffle=True)
# Looking at three points and labels
for i,(predictor,target) in enumerate(islice(train_dl,3)):
    print(target)
    print(predictor)

# Creating torch linear model from scratch 
torch.manual_seed(1)
# Initializing random weights and bias
weight=torch.randn(1,requires_grad=True)
bias=torch.zeros(1,requires_grad=True)
# @ operator used for matrix multiplication
def model(xb):
    return xb @ weight+bias
# using MSE as the loss function 
def loss_func(input,target):
    return ((target-input)**2).mean()
# Training the model 
learning_rate=0.001
num_epochs=200
log_epochs=10
for epoch in range(num_epochs):
    for x_batch,y_batch in train_dl:
        pred=model(x_batch)
        loss=loss_func(pred,y_batch)
        loss.backward()
    with torch.no_grad():
        weight-=weight.grad*learning_rate
        bias-=bias.grad*learning_rate
        weight.grad.zero_()
        bias.grad.zero_()
    if epoch %log_epochs==0:
        print(f' Epoch {epoch} Loss {loss.item():.4f}') 

# Extracting the results of the linear model 
print(f'Final Parameters {weight.item():.2f}, {bias.item():.2f}')
X_test=np.linspace(0,9,num=100,dtype=np.float32).reshape(-1,1)
X_test_norm=(X_test-np.mean(X_test))/np.std(X_test)
X_test_norm=torch.from_numpy(X_test_norm)
y_pred=model(X_test_norm).detach().numpy()
# Plotting the training examples and linear regression 
fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(X_train_std,Y,marker='^',s=18,c='green')
ax.plot(X_test_norm,y_pred,lw=2,linestyle='--')
ax.legend(['Training data','Linear regression'],loc='upper left',fontsize=15)
ax.set_xlabel('x',size=15)
ax.set_ylabel('y',size=15)
ax.tick_params(axis='both',which='major',labelsize=15)
plt.tight_layout()

# Creating new MSE function and stochastic gradient optimizer
loss_fn=nn.MSELoss(reduction='mean')
input_size=1
output_size=1
model=nn.Linear(input_size,output_size)
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
# Optimizing the linear model employing .step() method 
for epoch in range(num_epochs):
    for x_batch,y_batch in train_dl:
        # Generate predictions
        pred=model(x_batch)[:,0]
        # Calculate the loss
        loss=loss_fn(pred,y_batch)
        # Computing the gradients
        loss.backward()
        # Updating parameters using gradients
        optimizer.step()
        # Resetting gradients to 0 
        optimizer.zero_grad()
    if epoch%10==0:
        print(f'Epoch {epoch}, MSE {loss.item():.3f}')
# Printing the final parameters
print('Final parameters:',model.weight.item(),model.bias.item())

# Employing NN to classsify iris dataset 
iris=load_iris()
X=iris['data']
y=iris['target']
# Splitting the data into test and train sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1./3,random_state=1)
# Standardizing the training data; creating PyTorch dataset and Dataloader 
X_train_std=(X_train-X_train.mean())/X_train.std()
X_train_norm=torch.from_numpy(X_train_std).float()
y_train=torch.from_numpy(y_train)
train_ds=TensorDataset(X_train_norm,y_train)
torch.manual_seed(11)
train_dl=DataLoader(train_ds,batch_size=2,shuffle=True)

# Defining a two layer model built from two linear layers
class Model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        # Parent class constructor 
        super().__init__()
        self.layer1=nn.Linear(input_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,output_size)
    def forward(self,X):
        X=self.layer1(X)
        X=nn.Sigmoid()(X)
        X=self.layer2(X)
        X=nn.Softmax()(X)
        return X
# Defining parameters of the model 
input_size=X_train.shape[1]
hidden_size=16
output_size=3
model=Model(input_size,hidden_size,output_size)
# Specifying the optimizer
learning_rate=0.001
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
# Training the model 
num_epochs=100
loss_hist=[0]*num_epochs
accuracy_hist=[0]*num_epochs
for epoch in range(num_epochs):
    for x_batch,y_batch in train_dl:
        # Prediction by the model 
        prediction=model(x_batch)
        # Calculating the loss function
        loss=loss_fn(prediction,y_batch)
        # Calculating the gradients via backward propagation of the loss function
        loss.backward()
        # Updating parameters using optimizer here Adam algorithm 
        optimizer.step()
        # Setting gradients to zero
        optimizer.zero_grad()
        # Storing the loss history 
        loss_hist[epoch]+=loss.item()*y_batch.size(0)
        # Storing the accuracy history 
        is_correct=(torch.argmax(prediction,dim=1)==y_batch).float()
        accuracy_hist[epoch]+=is_correct.mean()
    loss_hist[epoch]/=len(train_dl.dataset)
    accuracy_hist[epoch]/=len(train_dl.dataset)

# Creating plots of accuracy and lost history 
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,8))
ax=ax.flatten()
ax[0].plot(range(1,101),loss_hist,lw=3)
ax[0].set_title('Training loss',size=15)
ax[0].set_xlabel('Training epoch',size=15)
ax[0].tick_params(axis='both',which='major',labelsize=15)
# Plotting accuracy history
ax[1].plot(range(1,101),accuracy_hist,lw=3)
ax[1].set_title('Accuracy',size=15)
ax[1].set_xlabel('Training epoch',size=15)
ax[1].tick_params(axis='both',which='major',labelsize=15)

# Evaluating the classification accuracy on the test dataset 
X_test_norm=(X_test-X_train.mean())/X_train.std()
# Creating the tensors from label and feature data
X_test_tensor=torch.from_numpy(X_test_norm).float()
y_test_tensor=torch.from_numpy(y_test)
# Predicting and determining the prediction accuracy 
prediction=model(X_test_tensor)
correct=(np.argmax(prediction.detach(),axis=1)==y_test_tensor).float()
accuracy=correct.mean()
print(f'The prediction accuracy {accuracy:3f}')

# Saving and reloading the model
path='iris_classsifier.pt'
torch.save(model,path)
# Loading the saved model 
model_new=torch.load(path,weights_only=False)
model_new.eval()
# Evaluating the model on the test dataset
pred_test=model_new(X_test_tensor)
correct=(np.argmax(pred_test.detach(),axis=1)==y_test_tensor).float()
accuracy=correct.mean()
print(f'Accuracy {accuracy:.3f}') 
# Saving only the learned parameters
path='iris_classifier_state.pt'
torch.save(model.state_dict(),path)
# To reload the model we need to construct the model as before and feed this parameters
model_new1=Model(input_size,hidden_size,output_size)
model_new1.load_state_dict(torch.load(path))

# Sigmoid function 
a = np.array([1, 1.4, 2.5]) ## first value must be 1
b = np.array([0.4, 0.3, 0.5])
# Calculating logistic activation 
def net_input(X,w):
    z=np.dot(w,X)
    return z
def log(z):
    return 1/(1+np.exp(-z))
def logistic_activation(X,w):
    val=net_input(X,w)
    res=log(val)
    return res
print(f'P(y=1|x)={logistic_activation(a,b):.4f}')

# W array with the shape (n_outputs,n_hidden units+1)
# The first column are the bias units
W = np.array([[1.1, 1.2, 0.8, 0.4],[0.2, 0.4, 1.0, 0.2],[0.6, 1.5, 1.2, 0.7]])
# A array with the shape (n_hidden_units+1,n_samples)
A = np.array([[1, 0.1, 0.4, 0.6]])
# Calculating logistic function
Z=np.dot(W,A[0])
y_probas=log(Z)
# Printing the input and result of logistic function 
print(f'Net input \n {Z}')
print(f'Logistic output \n {y_probas}') 
# To predict class label we could employ maximum probability
np.argmax(y_probas,axis=0)

# Softmax function 
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))
# Calculating probabilities
y_proba=softmax(Z)
np.sum(y_proba)
# Softmax in PyTorch 
torch.softmax(torch.from_numpy(Z),dim=0)

# Plotting sigmoid function and hyperbolic tangent 
def tanh(z):
    e_p=np.exp(z)
    e_m=np.exp(-z)
    return (e_p-e_m)/(e_p+e_m)
z=np.arange(-5,5,0.005)
log_act=log(z)
tanh_act=tanh(z)
# Plotting sigmoid and tangent hyperbolic functions
plt.plot(z,tanh_act,lw=3,linestyle='--',label='tanh')
plt.plot(z,log_act,lw=2,linestyle=':',label='sigmoid')
plt.ylim([-1.5,1.5])
plt.xlabel('net input $z$')
plt.ylabel('activation function $\phi(z)$')
plt.axhline(1,color='black',linestyle=':')
plt.axhline(0.5,color='black',linestyle=':')
plt.axhline(0,color='black',linestyle=':')
plt.axhline(-0.5,color='black',linestyle=':')
plt.axhline(-1,color='black',linestyle=':')
plt.legend(loc='lower left')
plt.tight_layout()
# Using tanh functions in NumPy and PyTorch 
np.tanh(z)
torch.tanh(torch.from_numpy(z))
# Logistic function 
from scipy.special import expit
expit(z)
# Using torch sigmoid as follows
torch.sigmoid(torch.from_numpy(z))
# ReLU for adressing the problem of vanishing gradients
torch.relu(torch.from_numpy(z))