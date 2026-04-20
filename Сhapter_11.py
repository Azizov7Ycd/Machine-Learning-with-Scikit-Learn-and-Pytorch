# Importing the necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
from neuralnet import NeuralMP

# Downlaoding MNIST dataset 
X,y=fetch_openml(name='mnist_784',version=1,return_X_y=True)
# Returning a Numpy Representation 
X=X.values 
y=y.astype(int).values
# X:70000 images with 784 pixels; y:70000 names of the pictures
print(X.shape)
print(y.shape)
# Nomalizing the pixels to range -1;1
X=((X/255)-0.5)*2
# Plotting the images 
fig,ax=plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax=ax.flatten()
y_un=np.unique(y)
for num,i in enumerate(y_un):
    im=X[y==num][0].reshape(28,28)
    ax[i].imshow(im,cmap='Greys') 
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# Plotting 25 7 characters
fig,ax=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax=ax.flatten()
for i in range(25):
    im=X[y==7][i].reshape(28,28)
    ax[i].imshow(im,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# Splitting the data into training, validation and test sets
X_tempt,X_val,y_tempt,y_val=train_test_split(X,y,test_size=10000,stratify=y,random_state=11)
X_train,X_test,y_train,y_test=train_test_split(X_tempt,y_tempt,test_size=5000,stratify=y_tempt,random_state=11)

# Initializing new instance of multilayer perceptron 
model1=NeuralMP(num_features=28*28,num_hidden=50,num_classes=10)


# Helper function to calculate sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
# Converting integer class labels into one-hot encoded labels 
def int_to_hot(y,num_labels): 
    ary=np.zeros((y.shape[0],num_labels))
    for i,val in enumerate(y):
        ary[i,val]=1
    return ary 

# Creating a function for splitting our data into minibatches
num_epochs=50
minibatch_size=100
def minibatch_generator(X,y,minibatch_size):
    indices=np.arange(X.shape[0]) # the indices we will be working with 
    np.random.shuffle(indices)
    for i in range(0,X.shape[0],minibatch_size):
        batch_idx=indices[i:i+minibatch_size]
        yield X[batch_idx],y[batch_idx]

# Testing the minibatch generator
for i in range(num_epochs):
    mini_bat=minibatch_generator(X_train,y_train,100)
    for X_mini_batch,y_mini_batch in mini_bat:
        print(X_mini_batch.shape)
        print(y_mini_batch.shape)  

# Defining MSE loss function 
def mse_loss(targets,probas,num_labels=10):
    onehot_targets=int_to_hot(targets,num_labels=num_labels)
    return np.mean((onehot_targets-probas)**2)
# Defining the resulting accuracy 
def accuracy(targets,predicted_labels):
    return np.mean(targets==predicted_labels)

# Calculating mean squared error via forward propagation 
_,probas=model1.forward(X_train)   # returning the values for out activation function 
mse=mse_loss(y_train,probas)
print(f'Initial mean square error is {mse:.2f}')

# Looking at the accuracy of the model 
predict_val=np.argmax(probas,axis=1)  # the class with the greatest probability 
acc=accuracy(y_train,predict_val)
print(f'The accuracy of the model is {acc*100:.2f}%')

# Computing mse in batch portions
def compute_mse_and_acc(nnet,X,y,num_labels=10,minibatch_size=100):
    mse,correct_pred,num_examples=0.,0,0
    minibatch_gen=minibatch_generator(X,y,minibatch_size)
    # iterating over each of the minibatch generators 
    for i,(features,targets) in enumerate(minibatch_gen):
        _,probas=nnet.forward(features)
        # determining the predicted labels and one-hot encoding the labels
        predict=np.argmax(probas,axis=1)
        one_hot_encoding=int_to_hot(targets,num_labels=10)
        # calculating loss and the number of correct prediction per batch
        loss=np.mean((probas-one_hot_encoding)**2)
        # the number of correctly predicted values, mean squared error as well as the total number of examples
        correct_pred+=(predict==targets).sum()  
        mse+=loss
        num_examples+=targets.shape[0]
    # calculating mse; accuracy 
    mse=mse/(i+1)
    acc=correct_pred/num_examples
    return mse,acc
# Determining the mean square error and accuracy on the validation data
mse,acc=compute_mse_and_acc(model1,X_val,y_val) 
# MSE on the validation set 
print(f'Initial validation MSE {mse:.1f}')
print(f'Initial validation accuracy {acc:.1f}')

# Training the model 
def train(model,X_train,y_train,X_valid,y_valid,num_epochs,learning_rate=0.1,minibatch_size=100):
    epoch_loss=[]
    epoch_train_acc=[]
    epoch_valid_acc=[]
    for e in range(num_epochs):
        # Creating batches to train the data on 
        minibatch_gen=minibatch_generator(X_train,y_train,minibatch_size)
        for X_train_mini,y_train_mini in minibatch_gen:
            # Computing the outputs of the model 
            a_h,a_out=model.forward(X_train_mini)
            # Computing the gradients
            d_loss_d_w_out,d_loss_d_b_out,d_loss_d_w_h,d_loss_d_b_h=model.backward(X_train_mini,a_h,a_out,y_train_mini)
            # Updating the weights
            model.weight_h-=learning_rate*d_loss_d_w_h
            model.bias_h-=learning_rate*d_loss_d_b_h
            model.weight_out-=learning_rate*d_loss_d_w_out
            model.bias_out-=learning_rate*d_loss_d_b_out
        # Epoch logging 
        train_mse,train_acc=compute_mse_and_acc(model,X_train,y_train)
        valid_mse,valid_acc=compute_mse_and_acc(model,X_valid,y_valid)
        train_acc=train_acc*100
        valid_acc=valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch {e+1:}/{num_epochs}')
        print(f'Mean Square Error {train_mse:.3f}')
        print(f'Train accuracy {train_acc:.3f}')
        print(f'Validation accuracy {valid_acc:.3f}')
    return epoch_loss,epoch_train_acc,epoch_valid_acc

# Training the neural network model 
np.random.seed(33)
epoch_losses,epoch_train_acc,epoch_valid_acc=train(model1,X_train,y_train,X_val,y_val,num_epochs=50,learning_rate=0.1)
# Plotting the mean square loss of the trained neural network 
plt.plot(range(len(epoch_losses)),epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Mean square error')
plt.tight_layout()
# Plotting training and validation accuracy
plt.plot(range(len(epoch_train_acc)),epoch_train_acc,label='Training')
plt.plot(range(len(epoch_valid_acc)),epoch_valid_acc,label='Validation')
plt.legend(loc='lower left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tight_layout()
# Evaluating the performance on the test dataset 
mse,acc=compute_mse_and_acc(model1,X_test,y_test)
print(f'The accuracy on the test dataset {acc:.2f}')
# Plotting first 25 misclassified examples
X_subset=X[:2500,:]
y_subset=y[:2500] 
# Determining the model predictions
_,probas=model1.forward(X_subset)  # the probabilities predicted by the model 
pred=np.argmax(probas,axis=1)
# Plotting first 25 misclassified labels 
misclasssified_samples=X_subset[pred!=y_subset][:25]
misclassified_labels=pred[pred!=y_subset][:25]
corr_labels=y_subset[pred!=y_subset][:25]
# Plotting the misclassified digits
fig,ax=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax=ax.flatten()
for i in range(25):
    # reshaping the data 
    img=misclasssified_samples[i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title(f'{i+1}' 
                    f'True {corr_labels[i]}\n' 
                    f'False {misclassified_labels[i]}')
ax[0].set_yticks([])
ax[0].set_xticks([])
plt.tight_layout() 