# Importing the necessary libraries
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Loading the iris dataset
iris=datasets.load_iris()
# Selecting petal width and petal length as feature; and class label as target 
X=iris.data[:,[2,3]]
y=iris.target
# Printing the class labels
print(f'Class labels {np.unique(y)}') 
# splitting the data into 30% test and 70% training data
# using stratify to keep the label proportion in both test and training data as in the original dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y,shuffle=True)
# Counting the number of occurences in both train and test dets
print('Labels counts in y',np.bincount(y))
print('Labels count in training set', np.bincount(y_train))
print('Labels count in test set', np.bincount(y_test))
# Conducting stadartization on the data
stan_scaler=StandardScaler()
stan_scaler.fit(X_train) # Fitting on the train data
X_train=stan_scaler.transform(X_train)  # Transforming both training and test sets using the same mean and standard deviation 
X_test=stan_scaler.transform(X_test)
# Using perceptron model from sklearn 
ppn=Perceptron(eta0=0.1,random_state=1) # eta parameter corresponds to the learning rate: if taken too large overshooting is possible if too small the convergence can be hardly achievable
ppn.fit(X_train,y_train)  # fitting the model 
# Conducting the predictions using the perceptron 
pred_val=ppn.predict(X_test)  # predicting target based on the test dataset 
print(f'The number of wrongly predicted values {sum(pred!=val for pred,val in zip(pred_val,y_test))}') 
# Computing the accuracy score of the perceptron model 
print('Accuracy: %.3f' %accuracy_score(y_test,pred_val))  # firstly true labels then predicted values
# Alternatively we can use .score method present in each classifier in sklearn to compute the model accuracy 
print('Accuracy: %.3f'%ppn.score(X_test,y_test)) # .score method combines predict and accuracy score in one function 

# Plotting the decision regions
def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
    markers=('o','s','^','v','<')
    colors=('green','blue','red','cyan','yellow')
    cmap=ListedColormap(colors[:len(np.unique(y))])  # assigning a color pro each label 
    # plotting the decision surface
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))  # Creating a meshgrid where xx1 are all x1 coordinates, xx2 are x2 coordinates
    lab=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)     # predicting labels based on coordinates of the grid
    lab=lab.reshape(xx1.shape)
    plt.contourf(xx1,xx2,lab,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    # plotting the scatter plots of points in x1,x2 coordinates
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],c=colors[idx],marker=markers[idx],label=f'Class{cl}',edgecolor='black')
    # highlightening test examples
    if test_idx:
        X_test,y_test=X[test_idx,:],y[test_idx]
        plt.scatter(x=X_test[:,0],y=X_test[:,0],c='none',edgecolors='black',alpha=1.0,linewidths=1,marker='o',s=100,label='Test set')

# Plotting the decision boundary as well as highlighting the test values
X_stacked=np.vstack((X_train,X_test))
y_stacked=np.hstack((y_train,y_test))
plot_decision_regions(X_stacked,y_stacked,classifier=ppn,test_idx=range(105,150))
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend(loc='upper left')
plt.tight_layout()

# Logistic Regression 
# Plotting the sigmoid function in the range [-7,7]
def sigmoid(x):
    return 1/(1+np.exp(-x))
plt.plot(np.arange(-7,7,0.01),sigmoid(np.arange(-7,7,0.01)))
plt.axvline(0.0,color='k')
plt.ylim(-0.1,1)
plt.xlabel('z')
plt.ylabel('$\sigma$')  # dollar signs imply LaTeX math formatting 
# y axis ticks and gridline
plt.yticks([0.0,0.5,1.0])
ax=plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
# Creating a plot that what show the loss function for different values of sigma
def loss_1(z):
    return -np.log(sigmoid(z))
def loss_0(z):
    return -np.log(1-sigmoid(z))
# Plotting the graphs
z=np.arange(-10,10,0.1)
sigma_z=sigmoid(z)
c1=[loss_1(x) for x in z]
c2=[loss_0(x) for x in z]
# Plotting the loss function if the true label is 1 against the sigmoid function 
plt.plot(sigma_z,c1,color='green',label='L(w,b) if true y is 1')
plt.plot(sigma_z,c2,color='blue',label='L(w,b) if true y is 0')
plt.xlim(0,1)
plt.ylim(0.0,5.1)
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w,b)')
plt.legend(loc='best')
plt.tight_layout()

# Creating a class for Logistic Regession 
# Changing the activation function to sigmoid activation function 
# The loss function was changed to negative log probability function 
class LogisticRegression:
    """
    Gradient based logistic regression
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
            loss=(-y.dot(np.log(output))-(1-y).dot(np.log(1-output)))
            self.loses_.append(loss)
        return self
    def net_input(self,X):
        return np.dot(X,self.w_)+self.b_
    def activation(self,z):
        return 1/(1+np.exp(np.clip(-z,-250,250)))  # clipping the values to the given interval 
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0.5,1,0)  
# Our logistic regression would work only for binary classification tasks 
# Selecting setosa and vesicolor flowers from the dataset 
X_train_subset=X_train[(y_train==0)|(y_train==1)]
y_train_subset=y_train[(y_train==0)|(y_train==1)] 
lg_reg=LogisticRegression(n_iter=1000,eta=0.3,random_state=1)
lg_reg.fit(X_train_subset,y_train_subset)
# Plotting decision boundary 
plot_decision_regions(X_train_subset,y_train_subset,classifier=lg_reg)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend(loc='upper left')
plt.tight_layout()

# Training multilabel logistic regression on the iris data 
lr=LogisticRegression(C=100.0,solver='lbfgs',multi_class='ovr')    
lr.fit(X_train,y_train) 
X_combined=np.concatenate([X_train,X_test])
y_combined=np.concatenate([y_train,y_test])
plot_decision_regions(X_combined,y_combined,classifier=lr,test_idx=range(105,150))
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend(loc="upper left")
plt.tight_layout()
# Predicting probabilities of class belonging 
lr.predict_proba(X_test[:3,:])
lr.predict_proba(X_test[:3,:]).sum(axis=1)
# To obtain  the class labels we can use argmax or predict from sklearn 
lr.predict(X_test[:3,:])
lr.predict_proba(X_test[:3,:]).argmax(axis=1)
# To predict the label of a single the additional dimension is needed
lr.predict(X_test[0,:].reshape(1,-1))
# Testing logistic regression with diffferent regularization parameters
# LogisticRegression.coef_ (n_classes,n_features)
weigths,params=[],[]
for c in np.arange(-5,5):
    lr=LogisticRegression(C=10.**c,multi_class='ovr')
    lr.fit(X_train,y_train)
    weigths.append(lr.coef_[0])  # appending coeffitients under different degree of regularization 
    params.append(10.**c)  # appending inverse regularization parameter
weights=np.array(weigths)
weights   #compute weights of two future for one color type (petal length,petal width)
plt.plot(params,weights[:,0],label='Petal length')
plt.plot(params,weights[:,1],label='Petal width',linestyle='--')
plt.ylabel('Weight Coefficient')
plt.xlabel('C')
plt.legend(loc='best')
plt.tight_layout()

# Support Vector Machines (SVM)
svc=SVC(C=1.0,random_state=1,kernel='linear')
svc.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=svc)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend(loc='upper left')
plt.tight_layout()

# Stochastic gradient descent classifier (SGDC)
ppn=SGDClassifier(loss='perceptron')
lr=SGDClassifier(loss='log')  # logistic regression 
svm=SGDClassifier(loss='hinge') # SVM
# Generating dataset for kernelized SVM model 
np.random.seed(9)
X_xor=np.random.randn(200,2)
# logic_xor returns True if compaed values differ
y_xor=np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor=np.where(y_xor,1,0)
y_xor
# Creating the scatterplots, signifying values corresponding to y=1 and y=0
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],color='royalblue',marker='o',label='Class 1')
plt.scatter(X_xor[y_xor==0,0],X_xor[y_xor==0,1],color='red',marker='s',label='Class 0')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.legend(loc='best')
plt.tight_layout()

# Creating a svm model with non-linear kernel 
svm=SVC(kernel='rbf',C=10.0,gamma=1.0,random_state=1)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()

# Changing the value of gamma 
svm1=SVC(random_state=1,kernel='rbf',C=1.0,gamma=0.2)
svm1.fit(X_combined,y_combined)
plot_decision_regions(X_combined,y_combined,classifier=svm1)
plt.xlabel('Petal length (standartized)')
plt.ylabel('Petal width (standartized)')
plt.legend(loc='upper left')
plt.tight_layout()
# Increading the value of gamma to 1
svm2=SVC(random_state=1,kernel='rbf',C=1.0,gamma=100)
svm2.fit(X_combined,y_combined)
plot_decision_regions(X_combined,y_combined,classifier=svm2)
plt.xlabel('Petal length (standartized)')
plt.ylabel('Petal width (standartized)')
plt.legend(loc='upper left')
plt.tight_layout()

# Decision trees 
# Entropy as a measure of impurity in decision trees in binary case
def entropy(p):
    return -p*np.log2(p)-(1-p)*np.log2(1-p)    # for the binary case
p=np.arange(0.0,1.0,0.01)
ent=[entropy(x) for x in p]   # Calling entropy for a range of proportions
plt.plot(p,ent)
plt.xlabel('Class membership probability p(i=1)')
plt.ylabel('Entropy')

# Plotting Entropy, Gini Index and Classification error for data for proportion in range from 0 to 1
def giniindex(p):
    return p*(1-p)+(1-p)*(1-(1-p))
def entropy(p):
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))
def classification_error(p):
    return 1-np.max([p,1-p])
x=np.arange(0,1,0.01)
# Computing the impurity metrics
entrop=[entropy(p) if p!=0 else None for p in x]
gini_index=[giniindex(p) for p in x]
classification_error_=[classification_error(p) for p in x]
sc_entropy=[e*0.5 if e else None for e in entrop]
# Plotting three plots for the metrics 
fig,ax=plt.subplots()
for i,lab,l_styl,c in zip([entrop,gini_index,classification_error_,sc_entropy],['Entropy','Gini Index','Classification Error','Scaled Entropy'],['-','--','--','-.'],['black','red','green','royalblue']):
    ax.plot(x,i,linestyle=l_styl,color=c,label=lab,lw=2) 
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.16),ncol=5,fancybox=True,shadow=True)  # bbox_to_anchor (x,y,width,height)
ax.axhline(y=1.0,lw=1,linestyle='--',color='k')
ax.axhline(y=0.5,lw=1,linestyle='--',color='k')
ax.set_xlabel('p(i=1)')
ax.set_ylabel('Impurity')
fig.tight_layout()

# Building a tree model from scikit learn 
tree_model=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree_model.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=tree_model)
plt.xlabel('Petal length')
plt.ylabel('Petal Width')
plt.legend(loc='upper left')
plt.tight_layout()

# Visualizing a decision tree
feature_names = ['Sepal length', 'Sepal width','Petal length', 'Petal width']
classes=['Setosa','Versicolor','Virginica']
tree.plot_tree(tree_model,feature_names=feature_names,filled=True,class_names=classes)

# Random Forest Classifier
ran_forest=RandomForestClassifier(n_estimators=25,random_state=1,n_jobs=2)
ran_forest.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=ran_forest)
plt.xlabel('Petal Length (standartized)')
plt.ylabel('Petal Width (standartized)')
plt.legend(loc='upper left')
plt.tight_layout()

# k nearest neighbors
knn=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=knn)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc='upper left')
plt.tight_layout()