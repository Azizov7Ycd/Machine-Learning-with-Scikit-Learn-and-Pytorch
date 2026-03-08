# Importing the necessary libraries
import numpy as np
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

# Creating a Perceptron class
class Perceptron:
    def __init__(self,eta=0.01,n_iter=50,random_state=1):  # Instance attributes
        self.eta=eta      # Learning rate
        self.n_iter=n_iter   # The number of iterations
        self.random_state=random_state  # Random number generator
    def fit(self,X,y):
        """
        Parameters: 
        X- [n_examples,n_features] predictors for perceptron training 
        y- [n_examples] target values 
        """
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1])  # corresponds to the number of predictors
        self.b_=np.float_(0.)
        self.errors_=[]
        for _ in range(self.n_iter):
            errors=0
            for xi,target in zip(X,y):  # for each sample and target variable
                update=self.eta*(target-self.predict(xi))
                self.w_+=update*xi  # each of the predictors' weights is being modeified
                self.b_+=update
                if update!=0:
                    errors+=1
            self.errors_.append(errors)
        return self
    def net_input(self,x):
        # returning a z value
        res=np.dot(x,self.w_)+self.b_
        return res
    def predict(self,x):
        return np.where(self.net_input(x)>=0,1,0) 
    
# Loading iris dataframe in Pandas and plotting 5 last lines
s = r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL',s)
# Reading the dataframe
df=pd.read_csv(s,header=None,encoding='utf-8')
df.tail()
# Selecting vesicolor and setosa target values
y=df.iloc[0:100,4].values 
y=np.where(y=='Iris-setosa',0,1)  # np.where(condition,x,y)
# Extract sepal length and petal length 
X=df.iloc[0:100,[0,2]].values
# Plotting 50 elements of the data
fig,ax=plt.subplots()
ax.scatter(x=X[:50,0],y=X[:50,1],color='red',marker='o',label='Setosa')
ax.scatter(x=X[50:,0],y=X[50:,1],color='blue',marker='^',label='Vesicolor')
ax.set_xlabel('Sepal length (cm)',size=18)
ax.set_ylabel('Petal length (cm)',size=18)
ax.legend(loc='upper left')

# Training the perceptron on the iris data
ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
fig,ax=plt.subplots(figsize=(4,4))
ax.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o',color='green')
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Number of updates')
ax.set_xticks(range(0,11,1))

# Visualizing boundary of two-dimensional dataset 
def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))]) 
    # plotting the decision surface
    x1_min,x1_max=X[:,0].min()-1,X[:,1].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))  # xx1 holds all x1 coordinates, xx2 holds all x2 coordinates
    lab=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[cl==y,0], y=X[cl==y,1], alpha=0.8,marker=markers[idx],color=colors[idx],label=f'Class {cl}',edgecolor='black')
# Plotting decision refions of perceptron 
plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal length (cm)')
plt.legend(loc='upper left')

# Adaline gradient descent 
class Adaline:
    """
    Parameters:
    eta: Learning rate between 0 and 1
    n_iter: The number of weight and bias updates
    random_number: Random number generator seed for random weight initialization
    Attributes:
    w_: weights after fitting 
    b_: bias unit after fitting 
    losses_: mean square loss function values in each epoch 
    """
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    def fit(self,X,y):
        """
        Parameters:
        X- training data, where rows are n_examples and columns are n_features
        y- target data [n_examples]
        """
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=X.shape[1])  # initializing random weights as small numbers of the normal distribution 
        self.b_=np.float_(0.)
        self.loses_=[]
        for i in range(self.n_iter):
            net_input=self.predict(X)  
            output=self.activation(net_input) # calculating the prediction based on the current weights
            errors=(y-output)          # conducting the substraction between two vectors 
            self.w_+=self.eta*2.0*X.T.dot(errors)/X.shape[0]
            self.b_+=self.eta*2.0*errors.mean()
            loss=(errors**2).mean()  # calculating the mean square error per iteration
            self.loses_.append(loss)
        return self
    def net_input(self,X):
        return np.dot(X,self.w_)+self.b_
    def activation(self,X):
        return X
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0.5,1,0)

# Plotting the loss against the number of epochs against different learning rates
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
ada1=Adaline(n_iter=15,eta=0.1).fit(X,y)
ax[0].plot(range(1,len(ada1.loses_)+1),np.log10(ada1.loses_),marker='o') 
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Log of MSE')
ax[0].set_title('Adaline-learning rate 0.1')
ada2=Adaline(n_iter=15,eta=0.001).fit(X,y)
ax[1].plot(range(1,len(ada2.loses_)+1),ada2.loses_,marker='s',color='green')
ax[1].set_ylabel('Log of MSE')
ax[1].set_xlabel('The number of epochs')
ax[1].set_title('Adaline- learning rate 0.001')

# Standardizing a dataset before applying Adaline classifier
# Standartizing the original data
X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
# Fitting Adaline with 20 epocjhs and eta==0.5
ada_std=Adaline(n_iter=20,eta=0.5)
ada_std.fit(X_std,y)

# Plotting decision regions for this classifier
plot_decision_regions(X_std,y,classifier=ada_std)
plt.title('Adaline-Gradient Descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
# Plotting mean square error versus each epoch 
plt.plot(range(1,len(ada_std.loses_)+1),ada_std.loses_,marker='^')
plt.ylabel('Mean square error')
plt.xlabel('The number of epochs')
plt.title('Standardized Adaline')
plt.tight_layout()

# Creating a class for stochastic gradient descent
class AdalineSDG:
    """
    Paremters:
    eta (learning rate): between 0.0 and 1.0
    n_iter: the number the training dataset is passed over
    shuffle: boolean; setting default as True (to prevent cycles)
    random_state: random weight generator for random weight initialization

    Attributes:
    w_: weights after ftting
    b_: bias unit after fitting
    lossess_: mean square loss function averaged for all training examples in each epoch 
    """
    def __init__(self,eta=0.01,n_iter=10,shuffled=True,random_state=1):  # Initializing the model 
        self.eta=eta
        self.n_iter=n_iter
        self.shuffled=shuffled
        self.random_state=random_state
        self.w_initialized=False
    def initialize_weights(self,s):  # initializing the weights and model bias
        self.rngen=np.random.RandomState(self.random_state)
        self.w_=self.rngen.normal(loc=0.0,scale=0.01,size=s)
        self.b_=np.float_(0.)
        self.w_initialized=True
    def shuffle(self,X,y):         # conducting the value permutation 
        r=self.rngen.permutation(len(y))
        return X[r],y[r]
    def update_weights(self,xi,target):
        output=self.activation(self.net_input(xi))
        error=target-output
        self.w_+=self.eta*2.0*error*xi
        self.b_+=self.eta*error*2.0
        loss=error**2
        return loss
    def net_input(self,X):
        return np.dot(X,self.w_)+self.b_
    def activation(self,X):  # compute linear activation
        return X
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0.5,1,0)
    def fit(self,X,y):
        self.initialize_weights(s=X.shape[1])
        self.losses_=[]
        for i in range(self.n_iter):
            if self.shuffled:  
                X,y=self.shuffle(X,y)  # Shuffling features and outcome if shuffle is True
            losses=[]
            for xi,target in zip(X,y):
                losses.append(self.update_weights(xi,target))
            avg_loss=np.mean(np.array(losses))
            self.losses_.append(avg_loss)
        return self
    def partial_fit(self,X,y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self.initialize_weights(size=X.shape[1])
        if y.ravel().shape[0]>1:
            for xi,target in zip(X,y):
                self.update_weights(xi,target)
        else:
            self.update_weights(X,y)
        return self 

# Plotting the decision regions of stochastic gradient descent algorithm 
ada_sgd=AdalineSDG(n_iter=15,eta=0.01,random_state=1)
ada_sgd.fit(X_std,y) 
# ploting the decision regions for stochastic gradient descent
plot_decision_regions(X_std,y,classifier=ada_sgd)
plt.title('Adaline-Stochastic Gradient descent')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend(loc='upper left')
plt.tight_layout()
# Plotting MSE against each epoch 
plt.plot(range(1,len(ada_sgd.losses_)+1),ada_sgd.losses_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()

