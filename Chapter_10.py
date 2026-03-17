# Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import cm 
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage 
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
# k-means clustering
X,y=make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=0)
plt.scatter(X[:,0],X[:,1],c='white',edgecolor='black',marker='o',s=70)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.tight_layout() 
# k-means clustering 
km=KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=44)
# predicting the cluster membership
y_km=km.fit_predict(X) 
# Plotting the scatter plot of clusters
plt.scatter(X[y_km==0,0],X[y_km==0,1],marker='s',c='lightgreen',s=50,edgecolor='black',label='Cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],marker='o',c='orange',edgecolor='black',s=50,label='Cluster 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],marker='v',c='lightblue',edgecolor='black',s=50,label='Cluster 3') 
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='*',c='red',edgecolor='black',s=250,label='Centroids')
plt.xlabel('Feature 1',fontsize=12)
plt.ylabel('Feature 2',fontsize=12)
plt.legend()
plt.grid()
plt.tight_layout()
# Elbow Method
# .inertia_ stores the sum of squared errors in all clusters
print(f'Distortion {km.inertia_:.2f}')
# storing the distortion while experimenting with different values of k 
disordens=[]
for k in range(2,11):
    km=KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=300,random_state=123)
    km.fit_predict(X)
    disordens.append(km.inertia_)
# Making an elbow plot 
plt.plot(range(2,11),disordens,marker='o')
plt.xlabel('The number of clusters',fontsize=12)
plt.ylabel('Sum of squared errors',fontsize=12)

# Silhouette plot
km=KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,random_state=44,tol=1e-4)
y_km=km.fit_predict(X)
cluster_labels=np.unique(y_km)
n_clusters=cluster_labels.shape[0]
# Computing the Silhouette values 
silhouette_vals=silhouette_samples(X,y_km,metric='euclidean')
y_ax_upper,y_ax_lower=0,0
yticks=[]
for i,c in enumerate(cluster_labels):
    c_silhuette_vals=silhouette_vals[y_km==c]
    c_silhuette_vals.sort()
    y_ax_upper+=len(c_silhuette_vals)
    color=cm.jet(float(i)/n_clusters)  # selecting the color from the color map
    plt.barh(range(y_ax_lower,y_ax_upper),c_silhuette_vals,height=1.0,edgecolor='None',color=color)
    # appending the position of in the yticks
    yticks.append((y_ax_lower+y_ax_upper)/2)
    y_ax_lower+=len(c_silhuette_vals)  # storing the position for the next cluster
silhouette_avg=np.mean(silhouette_vals) # computing the mean of all values
plt.axvline(silhouette_avg,linestyle='--',c='red') 
plt.yticks(ticks=yticks,labels=cluster_labels)
plt.xlabel('Silhuette Coefficient',fontsize=12)
plt.ylabel('Clusters',fontsize=12)
plt.tight_layout()
# Experimenting with clustering using 2 centroids
km=KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=123,tol=1e-4)
y_km=km.fit_predict(X)
plt.scatter(X[y_km==0,0],X[y_km==0,1],marker='s',c='orange',edgecolor='black',label='Cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],marker='v',c='lightgreen',edgecolor='black',label='Cluster 2')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='*',c='red',s=250,label='Centroids')
plt.xlabel('Feature 1',fontsize=12)
plt.ylabel('Feature 2',fontsize=12)
plt.legend(loc='lower left')
plt.grid()
plt.tight_layout()
# Creating Silhuette plot for this case too
cluster_labels=np.unique(y_km)
n_clusters=cluster_labels.shape[0]
# Computing the Silhouette values 
silhouette_vals=silhouette_samples(X,y_km,metric='euclidean')
y_ax_upper,y_ax_lower=0,0
yticks=[]
for i,c in enumerate(cluster_labels):
    c_silhuette_vals=silhouette_vals[y_km==c]
    c_silhuette_vals.sort()
    y_ax_upper+=len(c_silhuette_vals)
    color=cm.jet(float(i)/n_clusters)  # selecting the color from the color map
    plt.barh(range(y_ax_lower,y_ax_upper),c_silhuette_vals,height=1.0,edgecolor='None',color=color)
    # appending the position of in the yticks
    yticks.append((y_ax_lower+y_ax_upper)/2)
    y_ax_lower+=len(c_silhuette_vals)  # storing the position for the next cluster
silhouette_avg=np.mean(silhouette_vals) # computing the mean of all values
plt.axvline(silhouette_avg,linestyle='--',c='red') 
plt.yticks(ticks=yticks,labels=cluster_labels)
plt.xlabel('Silhuette Coefficient',fontsize=12)
plt.ylabel('Clusters',fontsize=12)
plt.tight_layout()

# Complete linkage clustering 
# Creating a toy dataset of 3 parameters and 10 observations
np.random.seed(123)
variables=['A','B','C']
labels=['ID_0','ID_1','ID_2','ID_3','ID_4']
# random_sample returns numbers in range [0,1]
X=np.random.random_sample([5,3])*10   
df=pd.DataFrame(X,columns=variables,index=labels)
df
# Computing the distance matrix
row_dist=pd.DataFrame(squareform(pdist(X,metric='euclidean')),columns=labels,index=labels)
# Computing a linkage matrix with complete linkage
row_clusters=linkage(df.values,method='complete',metric='euclidean')
# Creating a DataFrame
pd.DataFrame(row_clusters,columns=['row label 1','row label 2','distance','number of items in clusters'],index=[f'cluster{i+1}' for i in range(row_clusters.shape[0])])
# Plotting a dendrogram 
row_denf=dendrogram(row_clusters,labels=labels)
plt.ylabel('Euclidean distance')
plt.tight_layout()

# Combining dendrogram with a heat map
fig=plt.figure(figsize=(8,8),facecolor='white')
axd=fig.add_axes([0.09,0.1,0.2,0.6])
row_dend=dendrogram(row_clusters,orientation='left') 
# Returning the clustering lables
# Changing the observation in the order specified by clustering
df_rowclust=df.iloc[row_dend['leaves'][::-1]]
# Creating a heatmap 
axm=fig.add_axes([0.23,0.1,0.6,0.6])
cax=axm.matshow(df_rowclust,cmap='hot_r',interpolation='nearest')
# Adding a colorbar and assign feature and data record names
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels(['']+list(df_rowclust.columns))
axm.set_yticklabels(['']+list(df_rowclust.index))
plt.tight_layout()
# Agglomerative clustering in sklearn 
ac=AgglomerativeClustering(n_clusters=3,linkage='complete',metric='euclidean')
clust=ac.fit_predict(X)
print(f'The cluster labels {clust}')
# Doing agglomerative clustering with two samples
ac2=AgglomerativeClustering(n_clusters=2,linkage='complete',metric='euclidean')
clust=ac2.fit_predict(X)
print(f'The clusters labels {clust}')

# Generating moon samples 
X,y=make_moons(n_samples=200,noise=0.05,random_state=44)
plt.scatter(X[:,0],X[:,1],marker='s',c='lightblue',edgecolor='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
# Conducting k-means and agglomerative clustering on the data
fig,ax=plt.subplots(1,2,figsize=(8,4))
km=KMeans(n_clusters=2,init='k-means++',random_state=1)
y_km=km.fit_predict(X)
ax[0].scatter(X[y_km==0,0],X[y_km==0,1],marker='s',edgecolor='black',c='lightblue',label='Cluster 1')
ax[0].scatter(X[y_km==1,0],X[y_km==1,1],marker='o',edgecolor='black',c='red',label='Cluster 2')
ax[0].set_title('K-Means Clustering')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')
ac=AgglomerativeClustering(n_clusters=2,linkage='complete',metric='euclidean')
y_ac=ac.fit_predict(X)
ax[1].scatter(X[y_km==0,0],X[y_km==0,1],marker='s',edgecolor='black',c='lightblue',label='Cluster 1')
ax[1].scatter(X[y_km==1,0],X[y_km==1,1],marker='o',edgecolor='black',c='red',label='Cluster 2')
ax[1].set_title('Agglomerative Clustering')
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')
plt.legend()
plt.tight_layout()
# Plotting density based clustering 
fig,ax=plt.subplots(figsize=(5,5))
db=DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
y_db=db.fit_predict(X)
ax.scatter(X[y_db==0,0],X[y_db==0,1],marker='s',edgecolor='black',c='lightblue',label='Cluster 1')
ax.scatter(X[y_db==1,0],X[y_db==1,1],marker='o',edgecolor='black',c='red',label='Cluster 2')
ax.set_title('Density-based Clustering')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
plt.legend()
plt.tight_layout()


