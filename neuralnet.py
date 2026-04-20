# Importing the necessary libraries 
import numpy as np
from math import exp

# Helper function to calculate sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
# Converting integer class labels into one-hot encoded labels 
def int_to_hot(y,num_labels): 
    ary=np.zeros((y.shape[0],num_labels))
    for i,val in enumerate(y):
        ary[i,val]=1
    return ary 
# Creating a neural network class with only one hidden layer
class NeuralMP:
    def __init__(self,num_features,num_hidden,num_classes,random_seed=123):
        super().__init__()
        self.num_classes=num_classes
    # hidden layer
        rng=np.random.RandomState(random_seed)
        self.weight_h=rng.normal(loc=0.0,scale=0.1,size=(num_hidden,num_features))
        self.bias_h=np.zeros(num_hidden)
    # outer layer
        self.weight_out=rng.normal(loc=0.0,scale=0.1,size=(num_classes,num_hidden))
        self.bias_out=np.zeros(num_classes)
    # Forward propagation 
    # Input layer
    def forward(self,x):
        z_h=np.dot(x,self.weight_h.T)+self.bias_h
        a_h=sigmoid(z_h)
    # Output layer
        z_out=np.dot(a_h,self.weight_out.T)+self.bias_out
        a_out=sigmoid(z_out)
        return a_h,a_out
    # Backward Propagation 
    def backward(self,x,a_h,a_out,y):
        # one-hot encoding
        y_onehot=int_to_hot(y,self.num_classes)
        # dLoss/dOutWeights = dLoss/dOutAct*dOutAct/dOutNet*dOutNet/dOutWeight
        # where DeltaOut=dLoss/dOuAct*dOutAct/dOutNet
        # input/output dimensions
        d_loss_d_a_out=2.0*(a_out-y_onehot)/y.shape[0]
        # derivative of the sigmoid function
        d_a_out_d_z_out=a_out*(1-a_out)
        delta_out=d_loss_d_a_out*d_a_out_d_z_out
        # Gradient for output weights
        d_z_out_dw_out=a_h
        d_loss_dw_out=np.dot(delta_out.T,d_z_out_dw_out)
        d_loss_db_out=np.sum(delta_out,axis=0)
        # dLoss/dHiddenWeights=DeltaOut*dOutNet/dHiddenAct*dHiddenAct/dHiddenNet*dHiddenNet/dHiddenWeight
        d_z_out_a_h=self.weight_out
        d_loss_a_h=np.dot(delta_out,d_z_out_a_h)
        # derivative of the sigmoid function 
        d_a_h_d_z_h=a_h*(1-a_h)
        d_z_h_d_w_h=x
        d_loss_d_w_h=np.dot((d_loss_a_h*d_a_h_d_z_h).T,d_z_h_d_w_h)
        d_loss_d_b_h=np.sum((d_loss_a_h*d_a_h_d_z_h),axis=0)
        return (d_loss_dw_out,d_loss_db_out,d_loss_d_w_h,d_loss_d_b_h)

   

