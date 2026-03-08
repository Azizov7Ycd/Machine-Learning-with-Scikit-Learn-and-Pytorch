# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
# Creating a mock file 
csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df=pd.read_csv(StringIO(csv_data))
# Counting the missing data along the index (per column)
df.isnull().sum()
# .dropna can be used to drop rows or columns from the data
df.dropna(axis=0)
df.dropna(axis=1)
# dropping an array only if all values are NA
df.dropna(how='all')
df.dropna(thresh=4)  # how many non NA values to require
df.dropna(subset=['C'])  # dropn only rows where NA appears at specific columns 

# Imputation of the missing values
imp=SimpleImputer(missing_values=np.nan,strategy="mean")
imp.fit(df)
df_imputed=imp.transform(df.values)
df_imputed
# An alternative would be use Pandas fillna method
df.fillna(df.mean())

# Handling categorical variables in Pandas
df = pd.DataFrame([['green', 'M', 10.1, 'class2'],['red', 'L', 13.5, 'class1'],['blue', 'XL', 15.3, 'class2']])
df.columns=['color','size','price','classlabel']
df
# Mapping ordinal features
size_map={'XL':3,'M':2,'L':1}
df['size']=df['size'].map(size_map)
df 
# Using the inverse size dictionary 
inverse_size={v:k for k,v in size_map.items()}
df['size']=df['size'].map(inverse_size)
# Encoding the class labels of the dataframe
class_map={cl:idx for idx,cl in enumerate(np.unique(df['classlabel']))} 
df['classlabel']=df['classlabel'].map(class_map)
df # classlabel are encoded as numbers
# Mapping class labels back to original representations
reverse_class_map={v:k for k,v in class_map.items()}
df['classlabel']=df['classlabel'].map(reverse_class_map)
# LabelEncoder from scikitlearn 
lab_enc=LabelEncoder()
y=lab_enc.fit_transform(df['classlabel'].values) 
y # Label encoded with number values
# Inverse_transform can be used to transform labels to their original encoding 
lab_enc.inverse_transform(y) 

# Encoding categorical features as integers using LabelEncoder()
X=df[['color','size','price']].values
lab_enc=LabelEncoder()
X[:,0]=lab_enc.fit_transform(X[:,0])
# One hot encoding of categorical features
X=df[['color','size','price']].values
one_hot_encod=OneHotEncoder()
one_hot_encod.fit_transform(X[:,0].reshape(-1,1)).toarray()
# Using ColumnTrnasformer for selective column transformation 
X=df[['color','size','price']].values
c_transform=ColumnTransformer([('onehot',OneHotEncoder(),[0]),('nothing','passthrough',[1,2])]) 
c_transform.fit_transform(X).astype(float)

# pandas get_dummies function can be used for one hot encoding 
pd.get_dummies(df[['color','size','price']])
# drop_first parameter would allow droping one of the predictors to avoid multicollinearity 
pd.get_dummies(df[['color','size','price']],drop_first=True)

# Drop first parameter can be specified in scikit learn One Hod Encoder as well
one_hot=OneHotEncoder(categories='auto',drop='first')
c_transform=ColumnTransformer([('onehot',one_hot,[0]),('nothing','passthrough',[1,2])]) 
c_transform.fit_transform(df[['color','size','price']]).astype('float')
 # transforming dropping dropping the first column

# Encoding of categorical features
df = pd.DataFrame([['green', 'M', 10.1,'class2'],['red', 'L', 13.5,'class1'],['blue', 'XL', 15.3,'class2']])
df.columns=['color','size','price','classlabel']
df['x>M']=df['size'].apply(lambda x:1 if x in ['L','XL'] else 0)
df['x>l']=df['size'].apply(lambda x:1 if x in ['XL'] else 0)
del df['size']

# Preprocessing the wine dataset 
df = pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
# Looking at unique class labels
print(np.unique(df['Class label']))

# Splitting the data into train and test sets
X,y=df.iloc[:,1:].values,df.iloc[:,0]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.3,stratify=y)
# Normalization 
mms=MinMaxScaler()
X_train_norm=mms.fit_transform(X_train)
X_test_norm=mms.transform(X_test)

# Standartization and Normalization in NumPy 
ex = np.array([0, 1, 2, 3, 4, 5])
ex_norm=(ex-ex.min())/(ex.max()-ex.min())
ex_stan=(ex-ex.mean())/ex.std()
print('normalized',ex_norm)
print('standartized',ex_stan)
# Satandartisation in sklearn 
stan_scal=StandardScaler()
X_train_stan=stan_scal.fit_transform(X_train)
X_test_stan=stan_scal.transform(X_test)

# Applying regularized l1 logistic regression 
lr=LogisticRegression(penalty='l1',C=1.0,solver='liblinear',multi_class='ovr',random_state=1)
# Fitting the model 
lr.fit(X_train_stan,y_train)
print('Training accuracy',lr.score(X_train_stan,y_train))
print('Test accuracy',lr.score(X_test_stan,y_test))
# Returning the intercept_
lr.intercept_
lr.coef_
# Changing the regularization parameter C and plotting weights of the different values
fig,ax=plt.subplots()
colors = ['blue', 'green', 'red', 'cyan','magenta', 'yellow', 'black','pink', 'lightgreen', 'lightblue','gray', 'indigo', 'orange']
weights,params=[],[]
for c in range(-4,6):
    lr=LogisticRegression(penalty='l1',C=10.**c,solver='liblinear',multi_class='ovr',random_state=1)
    lr.fit(X_train_stan,y_train)
    weights.append(lr.coef_[0])  # selecting coef_ for the first class label
    params.append(10**c)
weights=np.array(weights)
for col,column in zip(colors,range(weights.shape[1])):
    ax.plot(params,weights[:,column],color=col,label=df.columns[column+1]) 
ax.axhline(0,linewidth=3,linestyle='--',color='black')
ax.set_ylabel('Weight coefficient')
ax.set_xlabel('Inverse Regularization Strength')
ax.set_xscale('log')
ax.set_xlim([-10**(-5),10**(5)])
ax.legend(loc='upper left',bbox_to_anchor=(1.38,1.03),ncols=1,fancybox=True,shadow=True)

# Initializing class for sequence backward selection (SBS)
class SBS:
    def __init__(self,estimator,k_features,scoring=accuracy_score,test_size=0.25,random_state=1):
        self.scoring=scoring
        self.estimator=clone(estimator)
        self.k_features=k_features
        self.test_size=test_size
        self.random_state=random_state
    def fit(self,X,y):
        # Splitting the dataset into train and test subsets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
        dim=X_train.shape[1]  # returning the number of features from the train data
        self.indices_=list(range(dim))
        self.subsets_=[self.indices_]
        self.scores_=[self._calc_score(X_train,X_test,y_train,y_test)]
    # Creating a loop getting read of the features one after the other
        while dim>self.k_features:
            scores=[]
            subsets=[]
            for p in combinations(self.indices_,r=dim-1):
                # calculating scores for different combinations of the columns
                score=self._calc_score(X_train[:,p],X_test[:,p],y_train,y_test)
                scores.append(score)
                subsets.append(p)
            # taking the best combination of predictors 
            best=np.argmax(scores)
            self.indices_=subsets[best] 
            self.subsets_.append(self.indices_)
            dim-=1
            self.scores_.append(scores[best])
        self.k_score=self.scores_[-1]
        return self
    def transform(self,X):
        return X[:,self.indices_]
    def _calc_score(self,X_train,X_test,y_train,y_test):
        self.estimator.fit(X_train,y_train)
        y_predict=self.estimator.predict(X_test)
        score=self.scoring(y_test,y_predict)
        return score
# Using sequantial backward selection to select one feature from knn classifier
knn=KNeighborsClassifier(n_neighbors=5)
sbs=SBS(knn,k_features=1)
sbs.fit(X_train_stan,y_train)
sbs.subsets_

# Plotting the classification accuracy 
k_feat=[len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.12])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout() 
plt.xticks(range(1,14))

# Looking at the 4 predictor values 
k4=[sbs.subsets_[9]]
df.columns[1:][k4]
# Assessing the performance on original dataset
knn.fit(X_train_stan,y_train)
knn.score(X_train_stan,y_train)
print(f'Training accuracy {knn.score(X_train_stan,y_train):.3f}')
print(f'Testing accuracy {knn.score(X_test_stan,y_test):.3f}')
# Assessing the performance using the subset 
knn.fit(X_train_stan[:,k4],y_train)
print(f'Training accuracy {knn.score(X_train_stan[:,k4],y_train):.3f}')
print(f'Testing accuracy {knn.score(X_test_stan[:,k4],y_test):.3f}')

# Assessing model performance with Random Forest 
feat_labels=df.columns[1:]
forest=RandomForestClassifier(random_state=1,n_estimators=500)
# Fitting the model
forest.fit(X_train,y_train)
# Retrieving feature importance
importances=forest.feature_importances_
# np.argsort for sorting in ascending order 
indices=np.argsort(importances)[::-1] # sorting the importance indices 
# printing the features and corresponting importances
for feat,val in zip(df.columns[1:],importances[indices]):
    print(f'{feat}:{val:.3f}')

# Creating a bar plot of feature importances 
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')  # placing in the descending order
plt.xticks(range(len(feat_labels)),feat_labels[indices],rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout() 

# Selecting features from the model 
sfm=SelectFromModel(forest,threshold=0.1,prefit=True)
X_selected=sfm.transform(X_train)
print('Number of features that meet this threshold',X_selected.shape[1])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))




