# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# Exploring the data 
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area','Central Air', 'Total Bsmt SF', 'SalePrice']
df=pd.read_csv(r'http://jse.amstat.org/v19n3/decock/AmesHousing.txt',sep='\t',usecols=columns)
df.head()
# Studying the data
df.info() # looking at non NA values
df.shape  # the shape of the data frame
# Converting string values of Central Air into integers
df['Central Air']=df['Central Air'].map({'Y':1,'N':0})
# Checking for NA values
df.isna().sum() 
df=df.dropna(how='any')
# Plotting the scatterplot matrix
scatterplotmatrix(df.values,names=df.columns,figsize=(10,10),alpha=0.5)
plt.tight_layout()
# Calculating correlation coeffients
cor_coef=np.corrcoef(df.values.T)  # each row should represent a variable each column an observation 
hm=heatmap(cor_coef,column_names=df.columns,row_names=df.columns)
plt.tight_layout()
plt.show()

# Creating a class to determine the linear regression coefficients using OLS
class LinearRegressionGD:
    def __init__(self,eta=0.01,n_iter=50,random_state=144):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state
    def fit(self,X,y):
        rgen=np.random.RandomState(self.random_state) # creating a random number generator
        # initializing starting weights 
        self.w_=rgen.normal(scale=0.01,loc=0.0,size=X.shape[1])   
        self.b_=rgen.normal(scale=0.01,loc=0.0,size=1)
        self.losses_=[]
        for i in range(self.n_iter):
            output=self.net_input(X)   # computing the prediction values
            error=(output-y)
            # updating the weights and bias
            self.b_+=-self.eta*2.0*error.mean()
            self.w_+=-self.eta*2.0*X.T.dot(error)/X.shape[0]
            loss=(error**2).mean()
            self.losses_.append(loss)
        return self
    # defining the function to calculate the prediction value
    def net_input(self,X):
        return np.dot(X,self.w_)+self.b_

# Using linear estimator for price prediction based on living area
x=df[['Gr Liv Area']].values
y=df[['SalePrice']].values
stand=StandardScaler()
x_std=stand.fit_transform(x)
y_std=stand.fit_transform(y).flatten()
# Fitting the OLS algorithm on the scaled data
lin_reg=LinearRegressionGD(eta=0.1)
lin_reg.fit(x_std,y_std) 
# plotting the loss function against each iteration 
plt.plot(range(1,lin_reg.n_iter+1),lin_reg.losses_)
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
# Plotting the scatterplots and regline on the data
plt.scatter(x_std,y_std,c='steelblue',s=70,edgecolors='white')
plt.plot(x_std,lin_reg.net_input(x_std),lw=2,color='black')
plt.xlabel('Living Area',fontsize=12)
plt.ylabel('The sale price',fontsize=12)

# inverse_transform method can be used to return transformed data into original scale
feature=np.array([[2500]])
scaler=StandardScaler()
feature_std=scaler.fit_transform(feature)
val_std=lin_reg.net_input(feature_std)[:,np.newaxis]
value=scaler.inverse_transform(val_std)
print(f'Sale price {value.flatten()[0]:.2f}')
# Printing the weights and intercept
print(f'The intercept {lin_reg.w_[0]:.3f}')
print(f'The slope coefficient {lin_reg.b_[0]:.3f}')

# Implementing linear regression in scikit learn 
lin_reg=LinearRegression()
lin_reg.fit(x,y)
# Predicting the values; depicting the slope and intercept
y_pred=lin_reg.predict(x)
print(f'Slope: {lin_reg.coef_.flatten()[0]:.3f}')
print(f'Intercept: {lin_reg.intercept_[0]:.3f}')
# Plotting the linear regression from sklearn 
# Plotting the scatterplots and regline on the data
plt.scatter(x,y,c='steelblue',s=70,edgecolors='white')
plt.plot(x,y_pred,lw=2,color='black')
plt.xlabel('Living Area',fontsize=12)
plt.ylabel('The sale price',fontsize=12)
plt.tight_layout()

# Doing RANSAC regression 
# min_samples the proportion of samples used for each iteration; residual_threshold=None implies using MAD in RANSAC 
ransac=RANSACRegressor(estimator=LinearRegression(),max_trials=100,min_samples=0.95,residual_threshold=65000,random_state=44)
ransac.fit(x,y)
# Plotting the inner and outer mask 
inlier_mask=ransac.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)
line_X=np.arange(1,10,1)
line_Y=ransac.predict(line_X[:,np.newaxis]) 
plt.scatter(x[inlier_mask],y[inlier_mask],c='steelblue',edgecolor='white',marker='o',label='Inliers')
plt.scatter(x[outlier_mask],y[outlier_mask],c='limegreen',edgecolor='white',marker='s',label='Outliers')
plt.plot(line_X,line_Y)
plt.xlabel('Living area above the ground in square feet')
plt.ylabel('Sale price in US dollars')
plt.legend(loc='upper left')
plt.tight_layout() 
# Printing the slope and the intercept of the RANSAC model 
# Plotting the scatterplots and regline on the data
print(f'Slope is {ransac.estimator_.coef_.flatten()[0]:.3f}')
print(f'Intercept is {ransac.estimator_.intercept_[0]:.3f}')
# defining the function to compute MAD
def mean_absol_dev(data):
    return np.mean(np.abs(data-np.mean(data)))
mean_absol_dev(y)

# Performance metrics for linear regression 
target='SalePrice'
y=df[target].values
x=df.loc[:,df.columns!=target].values
# Splitting the data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=44)
y_train=y_train[:,np.newaxis]
y_test=y_test[:,np.newaxis]
model=LinearRegression()
model.fit(X_train,y_train)
y_train_predict=model.predict(X_train)
y_test_predict=model.predict(X_test)

# Plotting the residual plot 
x_mi=np.min([np.min(y_test_predict),np.min(y_train_predict)])
x_ma=np.max([np.max(y_test_predict),np.max(y_train_predict)])
fig,ax=plt.subplots(1,2,figsize=(7,3),sharey=True)
ax[0].scatter(y_train_predict,y_train_predict-y_train,marker='s',c='limegreen',edgecolor='white',label='Training data')
ax[1].scatter(y_test_predict,y_test_predict-y_test,marker='o',c='steelblue',edgecolor='white',label='Test data')
ax[0].set_ylabel('Residuals')
for i in range(0,2):
    ax[i].set_xlabel('Predicted values')
    ax[i].legend(loc='upper left')
    # plotting horizontal line denoting postion of 0 
    ax[i].hlines(y=0,xmin=x_mi-100,xmax=x_ma+100,color='black',lw=2)
plt.tight_layout()

# Printing the mean_square_error
mse_y_train=mean_squared_error(y_train,y_train_predict)
mse_y_test=mean_squared_error(y_test,y_test_predict)
print(f'MSE train {mse_y_train:.3f}')
print(f'MSE test {mse_y_test:.3f}')
# Printing the mean_absolute_error
mae_train=mean_absolute_error(y_train,y_train_predict)
mae_test=mean_absolute_error(y_test,y_test_predict)
print(f'MAE train {mae_train:.3f}')
print(f'MAE test {mae_test:.3f}')

# Printing the coefficients of determination 
r2_train=r2_score(y_train,y_train_predict)
r2_test=r2_score(y_test,y_test_predict)
print(f'R^2 train {r2_train:.3f}')
print(f'R^2 test {r2_test:.3f}')

# Applying Ridge,LASSO and elastic net regressions
rige=Ridge(alpha=1.0)
lasso=Lasso(alpha=1.0)
elnet=ElasticNet(alpha=1.0,l1_ratio=0.5)

# Polynomial regression in sklearn 
x = np.array([ 258.0, 270.0, 294.0, 320.0, 342.0,368.0, 396.0, 446.0, 480.0, 586.0])[:,np.newaxis]
y = np.array([ 236.4, 234.4, 252.8, 298.6, 314.2,342.2, 360.8, 368.0, 391.2, 390.8])
lr=LinearRegression()
pr=LinearRegression()
# Adding a second degree polynomial term 
quadratic=PolynomialFeatures(degree=2)
X_quad=quadratic.fit_transform(x)

# Toy dataset to conduct prediction on 
X_new=np.arange(250,600,10)[:,np.newaxis]
# Fitting the simple linear regression model to the data
lr.fit(x,y)
y_pred_linear=lr.predict(X_new)
# Fitting quadratic regression on the data
pr.fit(X_quad,y)  
y_pred_quad=pr.predict(quadratic.fit_transform(X_new))
# Plotting the results
plt.scatter(x,y,label='Training points')
plt.plot(X_new,y_pred_linear,label='Linear fit',linestyle='--')
plt.plot(X_new,y_pred_quad,label='Quadratic fit')
plt.legend(loc='upper left')
plt.xlabel('Explanatory variable')
plt.ylabel('Predicted or known target values')
plt.tight_layout()

# Computing MSE and R^2 metrics while predicting on the data 
y_prlin=lr.predict(x)
y_prquad=pr.predict(X_quad) 
# Printing the mean square error
mse_lin=mean_squared_error(y,y_prlin)
mse_pol=mean_squared_error(y,y_prquad)
print(f'Training MSE linear {mse_lin:.3f},quadratic {mse_pol:.3f}')
# Printing the determination scores
r2_lin=r2_score(y,y_prlin)
r2_quad=r2_score(y,y_prquad)
print(f'The training r2 score linear {r2_lin:.3f},quadratic {r2_quad:.3f}')

# Fitting Polynomial Regression model to the Ames Housing data
x=df[['Gr Liv Area']].values
y=df[['SalePrice']].values
# Removing the outliers
x=x[(df['Gr Liv Area']<4000)]
y=y[(df['Gr Liv Area']<4000)] 

# Fitting the regression models
regr=LinearRegression()
# Creating quadratic and cubic features
quad=PolynomialFeatures(degree=2)
cubic=PolynomialFeatures(degree=3)
x_quad=quad.fit_transform(x)
x_cubic=cubic.fit_transform(x)
# Data to conduct prediction on 
X_fit=np.arange(x.min()-1,x.max()+1,1)[:,np.newaxis]
# Fitting the linear regression model and conducting the prediction 
regr.fit(x,y)
y_linpred=regr.predict(X_fit)
linear_r2=r2_score(y,regr.predict(x))  
# Fitting the quadratic model and computing r2 score
regr.fit(x_quad,y)
y_quadpred=regr.predict(quad.fit_transform(X_fit))
quadratic_r2=r2_score(y,regr.predict(x_quad))
# Fitting the cibic model 
regr.fit(x_cubic,y)
y_cubicpred=regr.predict(cubic.fit_transform(X_fit))
cubic_r2=r2_score(y,regr.predict(x_cubic)) 
# Plotting the results
plt.scatter(x,y,color='lightgray',label='Training points')
plt.plot(X_fit,y_linpred,label=f'Linear (d=1), $R^2$:{linear_r2:.2f}',color='blue',lw=2,linestyle=':')
plt.plot(X_fit,y_quadpred,label=f'Quadratic (d=2), $R^2$:{quadratic_r2:.2f}',color='red',lw=2,linestyle='-')
plt.plot(X_fit,y_cubicpred,label=f'Cubic (d=3), $R^2$:{cubic_r2:.2f}',color='green',lw=2)
plt.xlabel('Living area in square feet')
plt.ylabel('The sale price')
plt.legend(loc='upper left')
plt.tight_layout()
# Reusing the code with different x variable 
x=df[['Overall Qual']].values
y=df[['SalePrice']].values 

# Using Decision Tree Regressor 
x=df[['Gr Liv Area']].values
y=df['SalePrice'].values 
tree=DecisionTreeRegressor(max_depth=3)
tree.fit(x,y)
# Creating the regression plot 
def regression_plot(x,y,model):
    plt.scatter(x,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(x,model.predict(x),lw=2,color='black')
# Sorting the idx of ground area in the ascending order
sort_idx=x.flatten().argsort()
regression_plot(x[sort_idx],y[sort_idx],tree)
# Using RF algorithm for regression problem 
target='SalePrice'
predictors=df.columns[df.columns!=target]
X=df[predictors].values
y=df[target].values
# Splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
forest=RandomForestRegressor(n_estimators=1000,criterion='squared_error',random_state=123,n_jobs=-1)
forest.fit(X_train,y_train)
# Computing MAE metrics
mae_train=mean_absolute_error(y_train,forest.predict(X_train))
mae_test=mean_absolute_error(y_test,forest.predict(X_test))
print(f'MAE train {mae_train:.2f}')
print(f'MAE test {mae_test:.2f}')
# Computing the coefficient of determination 
r2_train=r2_score(y_train,forest.predict(X_train))
r2_test=r2_score(y_test,forest.predict(X_test))
print(f'R^2 Train {r2_train:.2f}')
print(f'R^2 Test {r2_test:.2f}') 
# Plotting the residual plot
y_train_predict=forest.predict(X_train)
y_test_predict=forest.predict(X_test)
x_mi=np.min([np.min(y_test_predict),np.min(y_train_predict)])
x_ma=np.max([np.max(y_test_predict),np.max(y_train_predict)])
fig,ax=plt.subplots(1,2,figsize=(7,3),sharey=True)
ax[0].scatter(y_train_predict,y_train_predict-y_train,marker='s',c='limegreen',edgecolor='white',label='Training data')
ax[1].scatter(y_test_predict,y_test_predict-y_test,marker='o',c='steelblue',edgecolor='white',label='Test data')
ax[0].set_ylabel('Residuals')
for i in range(0,2):
    ax[i].set_xlabel('Predicted values')
    ax[i].legend(loc='upper left')
    # plotting horizontal line denoting postion of 0 
    ax[i].hlines(y=0,xmin=x_mi-100,xmax=x_ma+100,color='black',lw=2)
plt.tight_layout()