# Importing the necessary libraries 
import torch 
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST 

# Creating a Python graph 
def compute_z(a,b,c):
    r1=torch.sub(a,b)
    r2=torch.mul(r1,2)
    r3=torch.add(r2,c)
    return r3 
# Creating PyTorch graphs with different tensors
print('Scalar inputs', compute_z(torch.tensor(1),torch.tensor(2),torch.tensor(3)))
print('Rank 1 inputs', compute_z(torch.tensor([1]),torch.tensor([2]),torch.tensor([3])))
print('Rank 2 inputs', compute_z(torch.tensor([[1]]),torch.tensor([[2]]),torch.tensor([[3]])))

# PyTorch tensor objects for storing and updating model parameters
a=torch.tensor(3.14,requires_grad=True)
b=torch.tensor([1.0,2.0,3.0],requires_grad=True)
print(a)
print(b)
# requires_grad_() can be run to set requires_grad to True
w=torch.tensor([3.5,6.3,-2.9])
print(w.requires_grad) 
w.requires_grad_()
print(w.requires_grad)

# Javier/Glorot initialization 
torch.manual_seed(1)
w=torch.empty(2,3)
nn.init.xavier_normal(w)
print(w)

# Defining the tensor objects inside the nn.Module class
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1=torch.empty(2,3,requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2=torch.empty(1,2,requires_grad=True)
        nn.init.xavier_normal_(self.w2)
# Automatic differentiation 
w=torch.tensor(1.0,requires_grad=True)
b=torch.tensor(0.5,requires_grad=True)
x=torch.tensor([1.4])
y=torch.tensor([2.1])
pred=torch.add(torch.mul(w,x),b)
loss=(y-pred).pow(2).sum()
# Backward propagation 
loss.backward()
print('dL/dw',w.grad)
print('dL/db',b.grad)
# Manually calculating the gradients
print(2*x*((w*x+b)-y))
print(2*((w*x+b)-y))

# Sequential module
model=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU())
# Configuring the model layers
nn.init.xavier_uniform_(model[0].weight)
l1_weight=0.01
l1_penalty=l1_weight*model[2].weight.abs().sum()
# SDG was used as optimizer; cross-entropy loss for binary classification 
loss_fn=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

# Solving XOR problem using the data
torch.manual_seed(11)
np.random.seed(44)
x=np.random.uniform(low=-1.0,high=1.0,size=(200,2))
y=np.ones(len(x))
y[x[:,0]*x[:,1]<0]=0
# Creating pyplot plots
fig,ax=plt.subplots(figsize=(6,6))
ax.scatter(x[y==0,0],x[y==0,1],marker='o',alpha=0.75,s=15)
ax.scatter(x[y==1,0],x[y==1,1],marker='>',alpha=0.75,s=15)
ax.set_xlabel(r'$x_1$',size=15)
ax.set_ylabel(r'$x_2$',size=15)
plt.tight_layout()
# Splitting the data into training and validation sets
n_train=100
x_train=torch.tensor(x[:n_train],dtype=torch.float32)
x_valid=torch.tensor(x[n_train:],dtype=torch.float32)
y_train=torch.tensor(y[:n_train],dtype=torch.float32)
y_valid=torch.tensor(y[n_train:],dtype=torch.float32)


# Employing logistic regression for classification 
log_reg=nn.Sequential(nn.Linear(2,1),nn.Sigmoid())
# Specifying loss function and an optimizer
loss_fn=nn.BCELoss()
optimizer=torch.optim.SGD(log_reg.parameters(),lr=0.001)     # Optimizing the parameters of the model  

# Creating dataset and dataloader
train_ds=TensorDataset(x_train,y_train)
batch_size=2
torch.manual_seed(1)
train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True)

# Creating a deeper model with more layers 
deep_log_reg=nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,4),nn.ReLU(),nn.Linear(4,1),nn.Sigmoid())
# Defining loss function and an optimizer
loss_fn=nn.BCELoss()
optimizer1=torch.optim.SGD(deep_log_reg.parameters(),lr=0.015) 
# Training the deeper model 
# Training during 200 epochs and recording the history of training epochs
torch.manual_seed(11) 
num_epochs=200
def train(model,train_dl,num_epochs,x_valid,y_valid):
    loss_hist_train=[0]*num_epochs
    accuracy_hist_train=[0]*num_epochs
    loss_hist_valid=[0]*num_epochs
    accuracy_hist_valid=[0]*num_epochs
    for epoch in range(num_epochs):
        for x_batch,y_batch in train_dl:
            pred=model(x_batch)[:,0]  # slices the single column from the prediction
            loss=loss_fn(pred,y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Storing the loss during the training process
            loss_hist_train[epoch] += loss.item()
            # Determining the number of correct predictions
            is_correct=((pred>=0.5).float()==y_batch).float()
            # Storing the accuracy during the training
            accuracy_hist_train[epoch]+=is_correct.mean().item()
        # Averaging loss and accuracy per epoch 
        loss_hist_train[epoch]/=n_train
        accuracy_hist_train[epoch]/=n_train/batch_size
        # Metrics on the validation dataset 
        pred=model(x_valid)[:,0]
        loss=loss_fn(pred,y_valid)
        loss_hist_valid[epoch]=loss.item()  # item method to obtain scalar value from a tensor
        is_correct=((pred>=0.5).float()==y_valid).float()
        accuracy_hist_valid[epoch]=is_correct.mean().item()
    return loss_hist_train,loss_hist_valid,accuracy_hist_train,accuracy_hist_valid

# Training the model 
train_loss,valid_loss,train_acc,valid_acc=train(log_reg,train_dl,num_epochs,x_valid,y_valid)

# Plotting loss and accuracy: train and validation 
fig,ax=plt.subplots(ncols=2)
ax=ax.flatten()
ax[0].plot(range(num_epochs),train_loss,lw=4)
ax[0].plot(range(num_epochs),valid_loss,lw=4)
ax[0].legend(['Train Loss','Validation Loss'],fontsize=15)
ax[0].set_xlabel('Epochs',size=15)
ax[1].plot(range(num_epochs),train_acc,lw=4)
ax[1].plot(range(num_epochs),valid_acc,lw=4)
ax[1].legend(['Train Accuracy','Validation Accuracy'],fontsize=15)
ax[1].set_xlabel('Epochs',size=15)
plt.tight_layout()  

# Training the  deeper model 
train_loss,valid_loss,train_acc,valid_acc=train(deep_log_reg,train_dl,num_epochs,x_valid,y_valid)

# Creating a model using the class module
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        l1=nn.Linear(2,4)
        a1=nn.ReLU()
        l2=nn.Linear(4,4)
        a2=nn.ReLU()
        l3=nn.Linear(4,1)
        a3=nn.Sigmoid()
        l=[l1,a1,l2,a2,l3,a3]
        self.module_list=nn.ModuleList(l)
    def forward(self,x):
        for f in self.module_list:
            x=f(x)
        return x
    # Defining a method for prediction 
    def predict(self,x):
        x=torch.tensor(x,dtype=torch.float32)
        pred=self.forward(x)[:,0]
        # returns 1 and 0 
        return (pred>=0.5).float()


# Defining an instance of new class and training it 
model=MyModule()
model  # The list of layers
# Defining loss function, optimizer and training the model 
loss_fn=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.015)
# Training the model 
train_loss,valid_loss,train_acc,valid_acc=train(model,train_dl,num_epochs,x_valid,y_valid)

# Plotting training loss/accuracy/decision regions
fig,ax=plt.subplots(ncols=3)
ax=ax.flatten()
ax[0].plot(range(num_epochs),train_loss,label='Train Loss')
ax[0].plot(range(num_epochs),valid_loss,label='Validation Loss')
ax[0].set_xlabel('Epochs')
ax[0].legend(fontsize=15)
ax[1].plot(range(num_epochs),train_acc,label='Train Accuracy')
ax[1].plot(range(num_epochs),valid_acc,label='Validation Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].legend(fontsize=15)
# Plotting the decision regions and true labels of the validation set
plot_decision_regions(X=x_valid.numpy(),y=y_valid.numpy().astype(np.int32),clf=model)
plt.xlabel(r'$x_1$',size=15)
plt.ylabel(r'$x_2$',size=15)

# Creating a custom class using nn.Module 
class NoisyLinear(nn.Module):
    # Constructing a custom class
    def __init__(self,input_size,output_size,noise_stdev=0.1):
        # Taking the properties of the parent class
        super().__init__()
        w=torch.Tensor(input_size,output_size)
        # nn.Parameter is a Tensor that is a module parameter; so gradients are updated automatically
        self.w=nn.Parameter(w)
        # Initializing the weights with Xavier Uniform distribution 
        nn.init.xavier_uniform_(self.w)
        # Initializing the bias of the model 
        b=torch.Tensor(output_size).fill_(0)
        self.b=nn.Parameter(b)
        self.noise_stdev=noise_stdev
    # Defining a forward method for that layer
    def forward(self,x,training=False):
        if training:
            noise=torch.normal(0.0,self.noise_stdev,x.shape)
        # Adding noise to the predictors for randomization 
            x_new=torch.add(x,noise)
        else:
            x_new=x
        return torch.add(torch.mm(x_new,self.w),self.b)

# Instantizing the layer and calling it three times on the input tensor
torch.manual_seed(11)
noisy_linear=NoisyLinear(4,2)
x=torch.zeros((1,4))
print(noisy_linear.forward(x,training=True))
print(noisy_linear.forward(x,training=True))
print(noisy_linear.forward(x,training=False)) 

# Defining a model for solving the XOR problem but implying the NoisyLinear as the first layer of the model 
class MyNoisyModel(nn.Module):
    def __init__(self):
        # parent class constructor
        super().__init__()
        self.l1=NoisyLinear(2,4)
        self.a1=nn.ReLU()
        self.l2=nn.Linear(4,4)
        self.a2=nn.ReLU()
        self.l3=nn.Linear(4,1)
        self.a3=nn.Sigmoid()
    # we need to define forward method while working with nn.Module
    def forward(self,x,training=False):
        x=self.l1(x,training)
        x=self.a1(x)
        x=self.l2(x)
        x=self.a2(x)
        x=self.l3(x)
        x=self.a3(x)
        return x
    def predict(self,x):
        x=torch.tensor(x,dtype=torch.float32)
        pred=self.forward(x)[:,0]  # Taking only the first column 
        return (pred>=0.5).float()
# Looking at the model 
torch.manual_seed(1)
model=MyNoisyModel()
model
# Training the model 
loss_fn=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.015)
# Training the model 
torch.manual_seed(1)
num_epochs=200
loss_hist_train=[0]*num_epochs
loss_hist_val=[0]*num_epochs
accuracy_hist_train=[0]*num_epochs
accuracy_hist_val=[0]*num_epochs
def train2(model,num_epochs,train_dl,x_valid,y_valid):
    for epoch in range(num_epochs):
        for x_batch,y_batch in train_dl:
            pred=model(x_batch,training=True)[:,0]  
            loss=loss_fn(pred,y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch]+=loss.item()
            is_correct=((pred>=0.5).float()==y_batch).float()
            accuracy_hist_train[epoch]+=is_correct.mean()
        # Calculating the average accuracy and loss per batch 
        loss_hist_train[epoch]/= 100/batch_size
        accuracy_hist_train[epoch]/= 100/batch_size
        # Calculating the loss and accuracy on the validation set 
        pred=model(x_valid)[:,0]
        loss=loss_fn(pred,y_valid)
        loss_hist_val[epoch]=loss.item()
        accuracy_val=((pred>=0.5).float()==y_valid).float()
        accuracy_hist_val[epoch]=accuracy_val.mean()
    return loss_hist_train,accuracy_hist_train,loss_hist_val,accuracy_hist_val 
# Training the model 
loss_train,accuracy_train,loss_val,accuracy_val=train2(model,num_epochs,train_dl,x_valid,y_valid)

# Plotting loss, accuracy and the decision borders
fig=plt.figure(figsize=(16,4))
ax=fig.add_subplot(1,3,1)
plt.plot(range(1,num_epochs+1),loss_train,lw=4)
plt.plot(range(1,num_epochs+1),loss_val,lw=4)
plt.legend(['Train Loss','Validation Loss'],fontsize=15)
ax.set_xlabel('Epochs',size=15)
ax=fig.add_subplot(1,3,2)
plt.plot(range(1,num_epochs+1),accuracy_train,lw=4)
plt.plot(range(1,num_epochs+1),accuracy_val,lw=4)
plt.legend(['Train Accuracy','Validation Accuracy'],fontsize=15) 
ax.set_xlabel('Epochs',size=15)
ax=fig.add_subplot(1,3,3)
plot_decision_regions(X=x_valid.numpy(),y=y_valid.detach().numpy().astype(np.int32),clf=model)
ax.set_xlabel(r'$x_1$',size=15)
ax.set_ylabel(r'$x_2$',size=15)
plt.tight_layout()

# Creating the model to predict MPG based on both categoric and numeric features
url = r'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names=['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
# Reading the dataframe
df=pd.read_csv(url,names=column_names,na_values='?',comment='\t',sep=" ",skipinitialspace=True)
df.isna().sum(axis=0)  # Summing up the values along the rows
# Removing the data with NA values
df=df.dropna(axis=0)
df=df.reset_index(drop=True)
# Splitting the data into train and test sets
df_train,df_test=train_test_split(df,train_size=0.8,shuffle=True,random_state=1)
# Describing the data statistics
df_train.describe().transpose()
numeric_column_names=['Cylinders','Displacement','Horsepower','Weight','Acceleration']
# Normalizing the train and test data
df_test_norm,df_train_norm=df_test.copy(),df_train.copy()
for col in numeric_column_names:
    df_test_norm[col]=(df_test_norm[col]-df_train[col].mean())/df_train[col].std()
    df_train_norm[col]=(df_train_norm[col]-df_train[col].mean())/df_train[col].std()
# Looking at the standartized values
df_test_norm.tail() 
# Buketizing the values for the year 
bundaries=torch.tensor([73,76,79],dtype=torch.int64)
v=torch.tensor(df_train_norm['Model Year'].values,dtype=torch.int64)
df_train_norm['Model Year Bucketed']=torch.bucketize(v,boundaries=bundaries,right=True)
v1=torch.tensor(df_test_norm['Model Year'].values,dtype=torch.int64)
df_test_norm['Model Year Bucketed']=torch.bucketize(v1,boundaries=bundaries,right=True)
# Appending the new column name to numeric_column_names
numeric_column_names.append('Model Year Bucketed')

# One hot encoding of categorical origin values
total_origin=len(set(df_train_norm['Origin']))
origin_encoded=one_hot(torch.from_numpy(df_train_norm['Origin'].values)-1) 
x_train_numeric=torch.from_numpy(df_train_norm[numeric_column_names].values)
# Concatenating the numeric and categoric columns
x_train=torch.cat([origin_encoded,x_train_numeric],1).float()
# Encoding the categoric values in the test set 
origin_encoded=one_hot(torch.from_numpy(df_test_norm['Origin'].values)-1)
x_test_numeric=torch.from_numpy(df_test_norm[numeric_column_names].values)
x_test=torch.cat([origin_encoded,x_test_numeric],1).float()

# Creating the label tensors
y_train=torch.tensor(df_train['MPG'].values).float()
y_test=torch.tensor(df_test['MPG'].values).float()

# Creating a dataloader from the train data
train_ds=TensorDataset(x_train,y_train)
torch.manual_seed(33)
batch_size=8
train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
# Building a model of 2 linear layers
x_train.shape[1]
model=nn.Sequential(nn.Linear(9,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU(),nn.Linear(4,1))
# Defining the loss function and otimizer
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
# Training the model for 200 epochs and displaying loss for 20 epochs
torch.manual_seed(1)
num_epochs=200
log_epochs=20
for epoch in range(num_epochs):
    loss_hist_train=0
    for x_batch,y_batch in train_dl:
        pred=model(x_batch)[:,0]
        loss=loss_fn(y_batch,pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist_train+=loss.item()
    if epoch%log_epochs==0:
        print(f'Epoch: {epoch}; Epoch Loss: {loss_hist_train:.3f}')

# Now feeding the test dataset into the model 
with torch.no_grad():
    pred=model(x_test)
    loss=loss_fn(y_test,pred)
    print(f'Mean Square Error (MSE): {loss:.3f}')
    print(f'Mean Average Error (MAE): {nn.L1Loss()(pred,y_test):.3f}')

# Classsification problem with the MNIST dataset
image_path=r"./"
transform=transforms.Compose([transforms.ToTensor()])
# Downloading train/test datasets
mnist_train_dataset=torchvision.datasets.MNIST(root=image_path,train=True,transform=transform,download=False)
mnist_test_dataset=torchvision.datasets.MNIST(root=image_path,train=False,transform=transform,download=False)
# Dataloader for the train data
torch.manual_seed(11)
train_dl=DataLoader(mnist_train_dataset,batch_size=64,shuffle=True)
# Plotting 10 images from the dataloader
fig,ax=plt.subplots(nrows=4,ncols=5)
ax=ax.flatten()
for batch_idx,(img,label) in enumerate(train_dl):
    for i in range(20):
        ax[i].imshow(img[i].squeeze(),cmap='gray')
        ax[i].set_title(label[i].item(),fontsize=18)
# Creating the classificational model 
mnist_train_dataset[0][0].shape # the first image [1,28,28] tensor
print(mnist_train_dataset[0][1])  # the label of the first image
# Model built of two linear layers
model=nn.Sequential(nn.Flatten(),nn.Linear(784,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,10),nn.Sequential())
# Specifying the loss function as well as the optimizer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# Training the model 
torch.manual_seed(1)
num_epochs=20
accuracy_hist_train=[0]*num_epochs
for epoch in range(num_epochs):
    accuracy=0
    for x_batch,y_batch in train_dl:
        pred=model(x_batch)
        loss=loss_fn(pred,y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Summing the correct prediction in each iteration
        is_correct=(np.argmax(pred,axis=1)==y_batch).float()
        accuracy+=is_correct.sum()
    accuracy_hist_train[epoch]=accuracy/len(train_dl.dataset)
# Reacing the accuracy of 98 %
print(accuracy_hist_train)
# Accuracy on the test dataset 

# Implementation of PyTorch Lightning
loss_fn=nn.CrossEntropyLoss()
class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self,img_shape=(1,28,28),hidden_units=(32,16)):
        super().__init__()
    # New PyTorchLightning attributes
        self.train_acc=Accuracy(task="multiclass",num_classes=10)
        self.valid_acc=Accuracy(task="multiclass",num_classes=10)
        self.test_acc=Accuracy(task="multiclass",num_classes=10)
        # Creating the model from a layer list 
        inputsize=img_shape[0]*img_shape[1]*img_shape[2]
        all_layers=[nn.Flatten()]
        # After flattening the image repeat several steps=> [Linear_Layer,ReLU() activation function]
        for hidden_unit in hidden_units:
            layer=nn.Linear(inputsize,hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            # The input to the next linear unit is has the shape of the hidden_units of the previous layer
            inputsize=hidden_unit
        # Appending the last linear layer and the softmax function to our putative model 
        all_layers.append(nn.Linear(hidden_units[-1],10))
        # all_layers.append(nn.Softmax())
        # Initializing the model using Sequential class
        self.model=nn.Sequential(*all_layers)
    # Defining the forward method 
    def forward(self,x):
        return self.model(x)
    # Defining the training procedure
    def training_step(self,batch,batch_idx):
        x,y=batch 
        logits=self(x)
        # Calculating the cross entropy loss
        loss=nn.CrossEntropyLoss(self(x),y)
        preds=torch.argmax(logits,axis=1)
        self.train_acc.update(preds,y)
        self.log('train loss',loss,prog_bar=True)
        return loss
    def on_train_epoch_end(self,outs):
        self.log("train_acc",self.train_acc.compute())
        self.train_acc.reset()
    # Defining validation step
    def validation_step(self,batch,batch_idx):
        x,y=batch 
        logits=self(x)
        loss=loss_fn(self(x),y)
        pred=torch.argmax(logits,dim=1)
        self.valid_acc.update(pred,y)
        self.log("valid_loss",loss,prog_bar=True)
        self.log("valid_acc",self.valid_acc.compute(),prog_bar=True,on_epoch=True)
        return loss
    # Defining test step
    def test_step(self,batch,batch_idx):
        x,y=batch
        logits=self(x)
        loss=loss_fn(logits,y)
        preds=torch.argmax(logits,axis=1)
        self.test_acc.update(preds,y)
        self.log("test_loss",loss,prog_bar=True)
        self.log("test_acc",self.test_acc.compute(),prog_bar=True)
        return loss
    # Configuring the optimizer
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=0.001)
        return optimizer

# LightningDataModule loading the data in PyTorch Lightning 
class MnistDataModule(pl.LightningDataModule):
    def __init__(self,data_path=r'./'):
        super().__init__()
        self.data_path=data_path
        self.transform=transforms.Compose([transforms.ToTensor()])
    def prepare_data(self):
        MNIST(root=self.data_path,download=True)
    def setup(self,stage=None):
        mnist_all=MNIST(root=self.data_path,train=True,transform=self.transform,download=False)
        self.train,self.val=random_split(mnist_all,[55000,5000],generator=torch.Generator().manual_seed(1))
        self.test=MNIST(root=self.data_path,train=False,transform=self.transform,download=False)
    def train_dataloader(self):
        return DataLoader(self.train,batch_size=64,num_workers=4)
    def val_dataloader(self):
        return DataLoader(self.val,batch_size=64,num_workers=4)
    def test_dataloader(self):
        return DataLoader(self.test,batch_size=64,num_workers=4)
# Initializing the data module
torch.manual_seed(1)
mnist_dm=MnistDataModule()
# Training the classifier
mnistclassifier=MultiLayerPerceptron()
if torch.cuda.is_available():
    trainer=pl.Trainer(max_epochs=10,gpus=1)
else:
    trainer=pl.Trainer(max_epochs=10)
# Training the model 
trainer.fit(model=mnistclassifier,datamodule=mnist_dm)








