# Importing the necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from use_functions import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# Loading the dataset 
df_wine=pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.info()
df_wine.head()
# Splitting the wine dataset and standartizing it to the unit variance
X,y=df_wine.iloc[:,1:],df_wine.iloc[:,0] 
# Splitting the data in the test and train datasets
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.3,random_state=1)
# Scaling the data
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)
# Computing the covariance matrix from the trained dataset
cov_mat=np.cov(X_test_scaled.T)  # the features should be located as rows
# computing eigenvalues and eigenvectors
eingenvals,eigenvecs=np.linalg.eig(cov_mat)
print('\nEigenvalues\n',eingenvals)

# Plotting the explained variance by each of the principal components
tot=sum(eingenvals)
var_exp=[round(i/tot,2) for i in sorted(eingenvals,reverse=True)]  # sorting the eigenvalues in the descending order
cum_var_exp=np.cumsum(var_exp)
plt.bar(range(1,14),var_exp,align='center',label='Individual expected variance')
plt.step(range(1,14),cum_var_exp,where='mid',label='Cumulative expained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Components')
plt.legend()
plt.xticks(range(1,14),['P'+str(i) for i in range(1,14)],rotation=45)
plt.tight_layout()

# Selecting the eigenvectors by decreasing order of eigenvalues
# Each row in the eigenvectorcorresponds to certain eigenvalue
eigen_pairs=[(np.abs(eingenvals[i]),eigenvecs[:,i]) for i in range(len(eingenvals))]
# Sorting based on the eigenvalues
eigen_pairs.sort(key=lambda k: k[0],reverse=True) 
# Stacking 2 eigenvectors capturing the most variance in the data 
w=np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
print('\nTransform matrix\n',w) 

# Transforming the training dataset onto principal components
X_train_pca=X_train_scaled.dot(w)
# Plotting the transformed dataset on the scatterplot 
colors=['r','b','g']
markers=['s','o','^']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1],color=c,marker=m,label=f'Class {l}')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='best')
plt.tight_layout()

# Importing a function for decision region plotting 
# Visualizing boundary of two-dimensional dataset 
pca=PCA(n_components=2)
l_reg=LogisticRegression(random_state=1,solver='lbfgs',multi_class='ovr')
# Dimensionality reduction 
X_train_pca=pca.fit_transform(X_train_scaled)
X_test_pca=pca.transform(X_test_scaled)
# Fitting the logistic regression model on the reduced dataset 
l_reg.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,classifier=l_reg)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout() 
# Plotting the logistic regression prediction on the transformed 
plot_decision_regions(X_test_pca,y_test,classifier=l_reg)
plt.legend(loc='lower left')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout() 
# explained_variance_ratio_ can be used to examine which proportion of variance each component corresponds to 
pca=PCA(n_components=None)
pca.fit_transform(X_train_scaled)
pca.explained_variance_ratio_

# Calculating the loadings of principal components
loadings=eigenvecs*np.sqrt(eingenvals)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']

# Plotting the loadings for the first principal component 
fig,ax=plt.subplots()
ax.bar(range(1,14),loadings[:,0],align='center')
ax.set_ylabel('Loadings of PC1')
ax.set_xticks(range(1,14))
ax.set_xticklabels(df_wine.columns[1:],rotation=90)
ax.set_ylim([-1,1])
plt.tight_layout()

# Using the loadings from fitted PCA
loadings=pca.components_.T*np.sqrt(pca.explained_variance_)
fig,ax=plt.subplots()
ax.bar(range(1,14),loadings[:,0],align='center')
ax.set_xticks(range(1,14))
ax.set_xticklabels(df_wine.columns[1:],rotation=90)
ax.set_ylim([-1,1])
ax.set_ylabel('Loading of PC1')
fig.tight_layout()

# Step 1: Computing the mean vector of the standartized data
mean_vectors=[]
for i in range(1,4):
    mean_vectors.append(np.mean(X_train_scaled[y_train==i],axis=0))
    print(f'MV{i}:{mean_vectors[i-1]}\n')
# Step 2: Computing within class and between class scatter matrices
d=13 # the number of features
S_W=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vectors):
    class_scater=np.zeros((d,d))
    for row in X_train_scaled[y_train==l]:
        row,mv=row.reshape(d,1),mv.reshape(d,1)
        class_scater+=(row-mv).dot((row-mv).T) 
    S_W+=class_scater
print(f'Within class scater matrix: \n {S_W.shape[0]}:{S_W.shape[1]}')
# Assumptions is made that the labels in the training set are uniformly distributed
print('Class label distribution',np.bincount(y_train)[1:])

# Computing the within matrices as correlation matrices
d=13 # the number of features
S_W=np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vectors):
    class_scater=np.cov(X_train_scaled[y_train==label].T)
    S_W+=class_scater
print(f'Scaled within class matrix {S_W.shape[0]}:{S_W.shape[1]}') 

# Step 3: Computing between class scatter matrices
mean_overall=np.mean(X_train_scaled,axis=0)  # calculate column along the indices
mean_overall=mean_overall.reshape(d,1) 

d=13
S_B=np.zeros((d,d))
for i,vector in enumerate(mean_vectors):
    n=X_train_scaled[y_train==i+1,:].shape[0]
    vector=vector.reshape(d,1)    # the mean vector of the certain class 
    S_B+=n*(vector-mean_overall).dot((vector-mean_overall).T)

print(f'Between class scatter matrix {S_B.shape[0]}:{S_B.shape[1]}')

# Step 4: Computing the eigenpairs of matrix S_W^(-1)S_B
eigen_vals,eigen_vecs=np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# Calculating eigenpairs
eigen_pairs=[(np.abs(eigen_vals)[i],eigen_vecs[:,i]) for i in range(len(eigen_vals))] 
# Sorting the values in the descending order of eigenvalues 
eigen_pairs.sort(key=lambda k: k[0],reverse=True)
# Printing the eigenvalues in the descending order
print('Egenvalues in the descending order: \n')
for eigen_pair in eigen_pairs:
    print(eigen_pair[0])

# Plotting the linear discriminants by decreasing eigenvalues
tot=sum(eigen_vals.real)
disc=[(i/tot) for i in eigen_vals.real]   # discriminability
cum_sum=np.cumsum(disc)
plt.bar(range(1,14),disc,align='center',label='Individual discriminability')
plt.step(range(1,14),cum_sum,where='mid',label='Cumulative discriminability')
plt.legend(loc='best')
plt.ylim([-0.1,1.1])
plt.tight_layout()

# Step 5: Creating a transformation matrix from the eigenvectors
w=np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real))
print('Transformation matrix \n',w)

# Step 6: Transforming the original dataset by multiplying it with transformation matrix
X_train_lda=X_train_scaled.dot(w)
colors=['royalblue','magenta','yellow']
markers=['o','s','^']
for lab,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==lab,0],X_train_lda[y_train==lab,1]*(-1),color=c,marker=m,label=f'Class {lab}')
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend()
plt.tight_layout()

# Implementing linear discriminant analysis in scikit-learn 
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train_scaled,y_train)  # Fitting using the label data as well 
# Fitting logisctic regression on the low dimensional training dataset
log_reg=LogisticRegression(multi_class='ovr',random_state=1,solver='lbfgs')
log_reg.fit(X_train_lda,y_train)
plot_decision_regions(X_train_lda,y_train,classifier=log_reg)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()

# Implementing LDA and Logistic Regression on test dataset 
X_test_lda=lda.transform(X_test_scaled) # fitting was already done on the training data
plot_decision_regions(X_test_lda,y_test,classifier=log_reg)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend()
plt.tight_layout()

# Applying t-SNE to visualize 64 dimensional dataset 
digits=load_digits()
# Plotting 4 first digits from the dataset 
fig,ax=plt.subplots(1,4)
for i in range(4):
    ax[i].imshow(digits.images[i],cmap='Greys')
# Using .data attribute we can get the digits in the tabular representation
digits.data.shape  # the datasets are represented by the rows and columns correspond to the pixels
# Assigning features and targets to 2 different variables
X_digits=digits.data
y_digits=digits.target
# Initializing tSNE object 
tsne=TSNE(n_components=2,init='pca',random_state=1)
X_digits_tsne=tsne.fit_transform(X_digits) 

# Plotting the t-SNE projections 
def plot_projections(X,train):
    fig,ax=plt.subplots()
    ax.set_aspect('equal')
    for i in range(10):                              # as therer are 10 digits in the dataset
        ax.scatter(X[train==i,0],X[train==i,1])
    for i in range(10):
        xtext=np.median(X[train==i,0])
        ytext=np.median(X[train==i,1])
        ax.text(xtext,ytext,str(i),fontsize=20,path_effects=([PathEffects.Stroke(linewidth=5, foreground="w"),PathEffects.Normal()])) 

plot_projections(X_digits_tsne,y_digits)




