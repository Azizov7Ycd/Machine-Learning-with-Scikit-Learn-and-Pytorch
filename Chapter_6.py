# Importing necessary librarries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import scipy as sc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import auc,roc_curve
from numpy import interp
from sklearn.utils import resample

# Reading the data and splitting it into train/test sets
df = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
# Now 30 feature columns to X
df.head()
X=df.iloc[:,2:].values
y=df.iloc[:,1].values
# Encoding the target variable using label encoding 
le=LabelEncoder()
y_fit=le.fit_transform(y)
le.classes_
# Splittting the data into test and training datasets
X_train,X_test,y_train,y_test=train_test_split(X,y_fit,stratify=y_fit,random_state=1,test_size=0.2) 
# Creating a pipelien consisting of StandardScaler, PCA and Logistic Regression 
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression())
# Using this model to predict both train and test datasets
pipe_lr.fit(X_train,y_train)
y_pred=pipe_lr.predict(X_test)
score=pipe_lr.score(X_test,y_test)
print(f'The prediction score is {score:.3f}') 

# Stratified k-fold cross validation with 10 folds
strat_cv=StratifiedKFold(n_splits=10)
k_folds=strat_cv.split(X_train,y_train)  # generating indices for train and test sets
# Calculating the accuracy scores based on each of the folds
scores=[]
for i,(train_fold,test_fold) in enumerate(k_folds):
    pipe_lr.fit(X_train[train_fold],y_train[train_fold])
    scores.append(pipe_lr.score(X_train[test_fold],y_train[test_fold]))
    print(f'k-fold {i+1},'
    f'Class distribution {np.bincount(y_train[train_fold])},'
    f'Accuracy {scores[i]:.3f}')
# Mean of the score values
mean=np.mean(scores)
std=np.std(scores)
print(f'\nCV Accuracy {mean:.3f} +- {std:.3f}')
# The other option is cross_val_score; where one can specify the features; labels and the number of cross_validations
scores=cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10)
print(f'The accuracy of the cross validation {np.mean(scores):.3f} +- {np.std(scores):.3f}')

# Creating a learning curve 
l_reg=make_pipeline(StandardScaler(),LogisticRegression(penalty='l2',max_iter=10000))
# Applying the learn_curve function 
train_sizes,train_scores,test_scores=learning_curve(estimator=l_reg,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1,10),cv=10)
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
# Plot the training data
plt.plot(train_sizes,train_mean,marker='s',color='red',markersize=8,label='Training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='red')
plt.plot(train_sizes,test_mean,marker='^',color='blue',linestyle='--',markersize=8,label='Test accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='blue')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.03])

# Determining train and test scores for different hyperparameter values
param_range=[10**(i) for i in range(-3,3)]
train_scores,test_scores=validation_curve(X=X_train,y=y_train,estimator=pipe_lr,param_name='logisticregression__C',param_range=param_range,cv=10) 
# Calculating mean and std of train and test scores
train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
# Plotting train and test cv results corresponding to different C parameters
plt.plot(param_range,train_mean,marker='o',color='royalblue',markersize=6,label='Training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='royalblue')
plt.plot(param_range,test_mean,marker='s',color='orange',markersize=6,label='Validation accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='orange')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8,1])

# Grid Search of the hyperparameters
pipe_svm=make_pipeline(StandardScaler(),SVC(random_state=1))
param_range=[10**(i) for i in range(-4,4)]
# Specifying the parameter grid 
# List of dictionaries specifying the paramters to look through
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}] 
gread_search=GridSearchCV(estimator=pipe_svm,param_grid=param_grid,scoring='accuracy',cv=10,refit=True,n_jobs=1)
# Fitting the gread_search pipeline
gread_search.fit(X_train,y_train)
# Printing best scores and best model parameters
print(gread_search.best_score_)
print(gread_search.best_params_)

# The best estimator is available as best_estimator parameter
clf=gread_search.best_estimator_
clf.fit(X_train,y_train)
print(f'Test score {clf.score(X_test,y_test):.3f}')

# Randomized search in scikit-learn 
param_range=[10**(i) for i in range(-4,4)]
# In RandomSearchCV the lists can be replaced with the distributions
param_range=sc.stats.loguniform(0.0001,1000.0)
# Drawing 10 random values from loguniform distribution 
param_range.rvs(10)

# Tuning the pipeline using RandomSearch algorithm 
pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))
# Making a parameter grid a list of dictionaries 
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__kernel':['rbf']}]
# Conducting randomized search 
rs=RandomizedSearchCV(estimator=pipe_svc,param_distributions=param_grid,scoring='accuracy',n_iter=20,cv=20,random_state=1,n_jobs=1)
# Fitting the model on the train dataset
rs.fit(X_train,y_train)
# Printing the parameters of the model 
print(rs.best_score_)
print(rs.best_params_)
# Scoring the prediction on the test data
pr=rs.best_estimator_
pr.score(X_test,y_test)  # 0.956 on the test data

# Conducting halving random search 
hs= HalvingRandomSearchCV(pipe_svc,param_distributions=param_grid,n_candidates='exhaust',resource='n_samples',factor=1.5,random_state=1,n_jobs=1)
hs.fit(X_train,y_train)
print(hs.best_params_)
print(hs.best_score_)
# Selecting the best classifier
clf=hs.best_estimator_  # the best estimator is already fitted on the data
print(f'{clf.score(X_test,y_test):.3f}')

# Nested cross-validation 
param_range=[10**(i) for i in range(-4,4)]
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]
# Building the inner loop using GridSearchCV
gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,cv=2,scoring='accuracy')  # finding the best estimator using cv=2
scores=cross_val_score(estimator=gs,X=X_train,y=y_train,cv=5)  # applying this best estimator on each of the 5 folds
# Printing the scores of the nested cross-validation 
print(f'Cross-validation accuracy is {np.mean(scores):.3f}+-{np.std(scores):.3f}')

# Comparing SupportVectorMachine pipeline to DecisionTreeClassifier 
gs=GridSearchCV(estimator=DecisionTreeClassifier(random_state=1),param_grid={'max_depth':[1,2,3,4,5,6,7,8,None]},cv=2,scoring='accuracy')  # the inner loop
scores=cross_val_score(estimator=gs,X=X_train,y=y_train,cv=5)
print(f'Cross-validation accuracy is {np.mean(scores):.3f}+-{np.std(scores):.3f}')

# Printing the confusion matrix using svc_pipeline predictor 
pipe_svc.fit(X_train,y_train)
y_pred=pipe_svc.predict(X_test)
conf_mat=confusion_matrix(y_true=y_test,y_pred=y_pred)  # printing the confusion matrix in this case
# Plotting the confusion matrix
fig,ax=plt.subplots(figsize=(3,3))
ax.matshow(conf_mat,cmap=plt.cm.Blues,alpha=0.3)
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        ax.text(x=j,y=i,s=conf_mat[i,j])
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Predicted class')
ax.set_ylabel('Actual class')

# Importing accuracy metrics
pre_val=precision_score(y_true=y_test,y_pred=y_pred)
print(f'The precision value is {pre_val:.3f}')
rec_val=accuracy_score(y_true=y_test,y_pred=y_pred)
print(f'The recall value {rec_val:.3f}')
f=f1_score(y_true=y_test,y_pred=y_pred) 
print(f'The f1-score is {f:.3f}')
mcc=matthews_corrcoef(y_true=y_test,y_pred=y_pred)
print(f'Matthew correlation coefficient {mcc:.3f}')

# Using GridSearchCV with 0 as positive class label and 'f1-score' as accuracy metrics
c_gamma=[10**(i) for i in range(-2,2)]
param_grid=[{'svc__kernel':['linear'],'svc__C':c_gamma},{'svc__kernel':['rbf'],'svc__C':c_gamma,'svc__gamma':c_gamma}]
scorer=make_scorer(score_func=f1_score,pos_label=0)
gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,cv=10,scoring=scorer)
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

# Plotting a ROC curve and calculating AUC
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(penalty='l2',random_state=1,solver='lbfgs',C=100.0))
X_train2=X_train[:,[4,14]]
#  Indices for train and test splits
cv=list(StratifiedKFold(n_splits=3).split(X_train,y_train))  # Indices for train and test splits
fig,axes=plt.subplots(figsize=(6,6))
mean_tpr=0.0
mean_fpr=np.linspace(0,1,100)
all_tpr=[]
for i,(train,test) in enumerate(cv):
    probas=pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])     # fitting the model on the train data; predicting the class probabilities on the test indices
    fpr,tpr,thresholds=roc_curve(y_train[test],probas[:,1],pos_label=1)
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=auc(fpr,tpr)
    axes.plot(fpr,tpr,label=f'ROC fold {i+1} (area={roc_auc:.3f})')
mean_tpr/=len(cv)
mean_tpr[-1]=1
mean_auc=auc(mean_fpr,mean_tpr)
axes.plot(mean_fpr,mean_tpr,'k--',label=f'Mean ROC area {mean_auc:.3f}',lw=2)
axes.plot([0,1],[0,1],linestyle='--',color='grey',label='Random guessing (area=0.5)')
axes.plot([0,0,1],[0,1,1],linestyle=':',color='black',label='Perfect performance (area=1.0)')
axes.set_xlim([-0.05,1.05])
axes.set_ylim([-0.05,1.05])
axes.set_xlabel('False positive rate')
axes.set_ylabel('True positive rate')
axes.legend(loc='lower right')

# The averaging method can be specified in the scorer function 
pre_scorer=make_scorer(score_func=precision_score,pos_label=1,greater_is_better=True,average='micro')

# Creating an imbalanced dataset 
X_imb=np.vstack([X[y_fit==0],X[y_fit==1][:40]])
y_imb=np.concatenate([y_fit[y_fit==0],y_fit[y_fit==1][:40]])
# The accuracy of the model which always predicts the majority class would be 90%
y_pred=np.zeros(y_imb.shape[0])
np.mean([y_pred==y_imb])*100  

# Resampling of the minority class
print('Number of class 1 examples before',y_imb[y_imb==1].shape[0])
X_resampled,y_resampled=resample(X_imb[y_imb==1],y_imb[y_imb==1],replace=True,n_samples=X_imb[y_imb==0].shape[0],random_state=123)
print('The number of class 1 examples after',X_resampled.shape[0])
# Comsequently the upsampled data can be stacked on the original data as follows
X_end=np.hstack((X_resampled,X_imb[y_imb==0]))
y_end=np.concatenate((y_resampled,y_imb[y_imb==0]))
# If taking always 0 model now only 50% accuracy achieved
y_pred=np.zeros(y_end.shape[0])
np.mean(y_pred==y_end)*100
