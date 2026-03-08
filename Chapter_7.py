# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator
from itertools import product
from scipy.special import comb
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# Creating a function calculating the error of ensemble classifier
def ensemble_error(n_classifier,error):
    k_start=int(math.ceil(n_classifier/2.))
    probs=[comb(n_classifier,k)*error**(k)*(1-error)**(n_classifier-k) for k in range(k_start,n_classifier+1)]
    return sum(probs)
# computing the ensemble error for previous example
ensemble_error(11,0.25)
# Plotting ensemble erros at different base errors
base_error=np.arange(0.0,1.0,0.1)
ensem_error=[ensemble_error(11,e) for e in base_error]
fig,ax=plt.subplots()
ax.plot(base_error,ensem_error,color='red',label='Ensemble error',linewidth=2)
ax.plot(base_error,base_error,color='blue',label='Base error',linewidth=2)
ax.set_xlabel('Base error')
ax.set_ylabel('Ensemble error')
ax.set_xticks(np.arange(0.0,1.25,0.25))
ax.set_yticks(np.arange(0.0,1.25,0.25))
ax.legend(loc='upper left')
ax.grid(alpha=0.5)

# The weighted ensemble learning 
np.argmax(np.bincount([0,0,1], weights=[0.2,0.2,0.6])) 
# Weighted ensemble learning with weights
ex=np.array([[0.9,0.1],[0.8,0.2],[0.4,0.6]])
aver=np.average(ex,axis=0,weights=[0.2,0.2,0.6])
np.argmax(aver)  

# MajorityVoteClassifier in Python 
class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers=classifiers
        self.named_classifiers={key:value for key,value in _name_estimators(classifiers)}
        self.vote=vote
        self.weights=weights
    def fit(self,X,y):
        if self.vote not in ('probability','classlabel'):
            raise ValueError(f'vote must be probability or classlabel; got {self.vote}')
        if self.weights and len(self.weights)!=len(self.classifiers):
            raise ValueError(f'The number of classifiers must be equal; got {len(self.classifiers)} classifiers and {len(self.weights)} weights')
        self.labelenc_=LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_=self.labelenc_.classes_  # the original classes of the label encoder
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf=clone(clf).fit(X,self.labelenc_.fit_transform(y))  # fitting each classifier on the transformed data
            self.classifiers_.append(fitted_clf)
        return self
    # Defining the predict method
    def predict(self,X):
        if self.vote=='probability':
            maj_vote=np.argmax(self.predict_proba(X),axis=1)
        else:    # class label vote
            # collecting results from clf.predict calls
            predictions=np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),axis=1, arr=predictions)  # determining the end prediction 
            maj_vote=self.labelenc_.inverse_transform(maj_vote) # transforming the output class back 
            return maj_vote
    # Definining predict_proba method for self.vote=='probability'
    def predict_proba(self,X):
        probas=np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        average_proba=np.average(probas,axis=0,weights=self.weights)
        return average_proba
    # getting parameters of the models
    def get_params(self,deep=True):
        if deep==False:
            return super().get_params(deep=False)
        else:
            out=self.named_classifiers.copy()
            for name,step in self.named_classifiers.items():
                for key,value in step.get_params(deep=True).items():
                    out[f'{name}__{key}']=value
            return out

# Preprocessing our dataset 
iris=datasets.load_iris()
# data for retrievieng the predictors; target for retrieving the results
X=iris.data[50:,[1,2]]  # using only 2 features
y=iris.target[50:]
# Label encoding of target variable
le=LabelEncoder()
y=le.fit_transform(y)
# Splitting the iris dataset into test and train subdivisions
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,stratify=y,random_state=123)  
# Building 3 classifiers and 2 pipelines
cl1=LogisticRegression(penalty='l2',C=0.001,solver='lbfgs',random_state=1)
cl2=DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=1)
cl3=KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')  # metric 2 means using euclidean distance
pipe1=Pipeline([['sc',StandardScaler()],['clf',cl1]]) # a list of lists contatining name of the step and estimator
pipe3=Pipeline([['sc',StandardScaler()],['clf',cl3]])
clf_labels=['Logistic Regression','Decision tree','KNN']
print('10-fold cross-validation:\n')
for label,clf in zip(clf_labels,[pipe1,cl2,pipe3]):
    scores=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print(f'ROC_AUC: {scores.mean():.2f} +/- {scores.std():.2f},{label}')

# Combining the 3 classifiers using majority vote
clf_ensemble=MajorityVoteClassifier(classifiers=[pipe1,cl2,pipe3])
clf_labels+=['Majority Voting']
for label,clf in zip(clf_labels,[pipe1,cl2,pipe3,clf_ensemble]):
    scores=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print(f'ROC_AUC {scores.mean():.2f} +- {scores.std():.2f} {label}')  
# Implememting the ensemble classifier on the test data
colors=['black','orange','red','green']
styles=['-.',':','--','-']
all_clf=[pipe1,cl2,pipe3,clf_ensemble]
for clf,label,color,style in zip(all_clf,clf_labels,colors,styles):
    y_pred=clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    fpr,tpr,threshold=roc_curve(y_test,y_pred)
    roc_auc=auc(x=fpr,y=tpr)  # the area under the ROC curve
    plt.plot(fpr,tpr,color=color,linestyle=style,label=f'{label} AUC:{roc_auc:.2f}')
plt.plot([0,1],[0,1],linestyle='--',color='gray',linewidth=2)
plt.legend()
plt.ylabel('True positive rate (TPR)')
plt.xlabel('False positive rate (FPR)')
plt.grid(alpha=0.5)
plt.xlim([-0.1,1.1]) 
plt.ylim([-0.1,1.1])

# Plotting the decision regions
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train) 
x_min=X_train_std[:,0].min()-1
x_max=X_train_std[:,0].max()+1
y_min=X_train_std[:,1].min()-1
y_max=X_train_std[:,1].max()+1
# np.meshgrid function used to create the cartesian product of coordinates
x1,y1=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))  # cartesian product of both arays: x and y coordinates stored separately 
fig,ax=plt.subplots(ncols=2,nrows=2,sharex='col',sharey='row',figsize=(7,5))
for idx,clf,lab in zip(product([0,1],[0,1]),all_clf,clf_labels):
    clf.fit(X_train_std,y_train)
    Z=clf.predict(np.c_[x1.ravel(),y1.ravel()])  # making predictions on each point of the meshgrid
    Z=Z.reshape(x1.shape)  # Reshaping the data as x1 coordinates of the meshgrid
    ax[idx[0],idx[1]].contourf(x1,y1,Z,alpha=0.3)
    ax[idx[0],idx[1]].scatter(X_train_std[y_train==0,0],X_train_std[y_train==0,1],c='blue',marker='^',s=50)
    ax[idx[0],idx[1]].scatter(X_train_std[y_train==1,0],X_train_std[y_train==1,1],c='green',marker='s',s=50)
    ax[idx[0],idx[1]].set_title(lab)
plt.text(-3.5,-5,s='Sepal width [standartized]',ha='center',va='center',fontsize=12)
plt.text(-12.5,4.5,s='Sepal length [standartized]',ha='center',va='center',fontsize=12,rotation=90)
# Getting the parameters from Majorityvote object 
clf_ensemble.get_params()
# Conducting the GridSearchCV on the tree depth and regularization of the tree classifier
params={'pipeline-1__clf__C':[0.001,0.01,0.1,1,10],'decisiontreeclassifier__max_depth':[1,2]}
gr_cv=GridSearchCV(estimator=clf_ensemble,param_grid=params,cv=10,scoring='roc_auc')
gr_cv.fit(X_train_std,y_train)
# Printing 10 ROC-AUC curves and the respective parameters
for r,_ in enumerate(gr_cv.cv_results_['mean_test_score']):
    mean_score=gr_cv.cv_results_['mean_test_score'][r]   # the mean score for each hyperparameter combination
    std_dev=gr_cv.cv_results_['std_test_score'][r]  # the standard deviation for each hyperparameter combination 
    params=gr_cv.cv_results_['params'][r]   # the hyperparameter combination tested
    print(f'{mean_score:.3f} +- {std_dev:.3f}; {params}')
print(f'{gr_cv.best_params_}') # best parameters for the cross validation
print(f'Best ROC-AUC {gr_cv.best_score_:.2f}') # the best result for cross validation 

# Importing the wine data dataset 
wine=datasets.load_wine()
X=pd.DataFrame(wine.data)
y=pd.DataFrame(wine.target)
X.columns=['Alcohol','Malic acid', 'Ash','Alcalinity of ash','Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
y.columns=['Class label']
# Filtering out the predictor variables and target data 
X=X.loc[y['Class label']!=1,['Alcohol','OD280/OD315 of diluted wines']].values  # selecting 2 columns from the dataset 
y=y.loc[y['Class label']!=1,:].values  # dropping all classes apart from the first one
# Conducting the label encoding 
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)
# Building a bagging classifier
tree=DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=1)
bag=BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,bootstrap=True,random_state=1,n_jobs=1)
# Comparing the accuracy score of unprudent tree versus bagging classifier 
tree.fit(X_train,y_train)
# Returning the training and test accuracy for the unprudent tree
print(f'Test Accuracy: {tree.score(X_test,y_test):.3f}')
print(f'Train Accuracy {tree.score(X_train,y_train):.3f}')
# Comparing the accuracy of the bagging algorithms
bag.fit(X_train,y_train)
print(f'Train accuracy {bag.score(X_train,y_train):.3f}')
print(f'Test accuracy {bag.score(X_test,y_test):.3f}')

# Conveying the decision bondary of unprudent tree and bagging classifier
x_min=X[:,0].min()-1
x_max=X[:,0].max()+1
y_min=X[:,1].min()-1
y_max=X[:,1].max()+1
# Creating a meshgrid for contour plot
x1,y1=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
fig,axes=plt.subplots(nrows=1,ncols=2,sharex='col',sharey='row',figsize=(8,3))
for i,clf,label in zip([0,1],[tree,bag],['Tree classifier','BaggingClassifier']):
    clf.fit(X_train,y_train)
    Z=clf.predict(np.c_[x1.ravel(),y1.ravel()])
    Z=Z.reshape(x1.shape)
    axes[i].contourf(x1,y1,Z,alpha=0.3)
    # ploting scatter plot for two classes
    axes[i].scatter(X_train[y_train==0,0],X_train[y_train==0,1],marker='^',color='blue',label=label)
    axes[i].scatter(X_train[y_train==1,0],X_train[y_train==1,1],marker='o',color='green',label=label)
    axes[i].set_title(label)
axes[0].set_ylabel('Alcohol',fontsize=12)
plt.tight_layout()
plt.text(0.0,-0.2,s='OD280/OD315 of diluted wines',fontsize=16,ha='center',va='center',transform=axes[1].transAxes)

# Adaboost (Adaptive boosting)
# Epsilon (weighted error computation) 
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])    # true values
yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])  # predicted values
correct=(y==yhat) 
weights=np.full(10,0.1)  # the initial weights normalized to 1
epsilon=np.dot(weights,~correct) 
# Computing the coefficient alpha
alpha_j=0.5*np.log((1-epsilon)/epsilon)
alpha_j #~0.424
# updating the weights
update_if_correct=0.1*np.exp(-alpha_j*1*1)  # if prediction and the true value is identical 
update_if_wrong=0.1*np.exp(-alpha_j*1*(-1))  # if prediction and true value are not the same
# Updating new weights
weights=np.where(correct,update_if_correct,update_if_wrong) 
print(weights)
# Normalizing weights to 1
normalized_weights=weights/np.sum(weights) 
print(normalized_weights) 
# Comparing AdaBoost with simple decision stump
tree=DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=1)
ada_boost=AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)
# Fitting both models
tree.fit(X_train,y_train)
ada_boost.fit(X_train,y_train)
# Printing the respective accuracy scores on train and test data
# For tree stump
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)
print(f'The accuracy ratio train vs test {accuracy_score(y_train,y_train_pred):3f}\{accuracy_score(y_test,y_test_pred):.3f}')
# For AdaBoost
y_train_pred=ada_boost.predict(X_train)
y_test_pred=ada_boost.predict(X_test)
print(f'The accuracy ratio train vs test{accuracy_score(y_train,y_train_pred):.3f}\{accuracy_score(y_test,y_test_pred)}')
# Plotting the decision regions of Adaboost 
x_min=X_train[:,0].min()-1
x_max=X_train[:,0].max()+1
y_min=X_train[:,1].min()-1
y_max=X_train[:,1].max()+1
# Building the meshgrid for countorplot
x1,y1=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(6,8),sharex='col',sharey='row')
for i,title,clf in zip([0,1],['Decision Stump','AdaBoost'],[tree,ada_boost]):
    clf.fit(X_train,y_train)
    Z=clf.predict(np.c_[x1.ravel(),y1.ravel()])
    Z=Z.reshape(x1.shape)
    axes[i].contourf(x1,y1,Z,alpha=0.4)
    axes[i].scatter(X_train[y_train==0,0],X_train[y_train==0,1],marker='o',color='red')
    axes[i].scatter(X_train[y_train==1,0],X_train[y_train==1,1],marker='^',color='blue')
    axes[i].set_title(title,fontsize=12)
    axes[i].set_ylabel('Alcohol',fontsize=12)
plt.text(-0.2,-0.2,s='OD280/OD315 of diluted wines',fontsize=12,ha='center',va='center',transform=axes[1].transAxes)
plt.tight_layout()

# Applying XGBoost Classifier
XGboost=xgb.XGBClassifier(n_estimators=1000,learning=0.01,max_depth=4,random_state=1,use_label_encoding=False)
# Model fitting 
XGboost.fit(X_train,y_train)
# Conducting the prediction on test data
y_predtrain=XGboost.predict(X_train)
y_predtest=XGboost.predict(X_test)
# Printing the accuracy score for train and test data 
gbm_train=accuracy_score(y_train,y_predtrain)
gbm_test=accuracy_score(y_test,y_predtest)
# Printing XGBoost train and test accuracies
print(f'XGBoost train/test accuracies {gbm_train:.3f}/{gbm_test:.3f}') 