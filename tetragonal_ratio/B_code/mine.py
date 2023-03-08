#! /usr/bin/env python

from matminer.featurizers.conversions import CompositionToOxidComposition, StrToComposition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, ShuffleSplit, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
def readdata(filename):
    df = pd.read_csv(filename,sep=',')
    return df
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def RF(X,y):
    model = RandomForestRegressor(n_estimators=50, random_state=1)
    model.fit(X, y)
    return model
def cv_scores(model,X,y):
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
    rmse_scores = [np.sqrt(abs(s)) for s in scores]
    CV_R2=np.mean(np.abs(r2_scores))
    CV_RMSE=np.mean(np.abs(rmse_scores))
    return CV_R2,CV_RMSE
def GPR(X,y):
    kernel = C(1.0,(1e-3, 1e3))*RBF(10,(1e-2,1e2))
    model = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
    model.fit(X,y)
    return model
def train_scores(model,X,y):
    y_pred=model.predict(X)
    MSE = metrics.mean_squared_error(y, y_pred)
    R2 = metrics.r2_score(y, y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y, y_pred))
    MAE = metrics.mean_absolute_error(y, y_pred)
    MAPE=mape(y, y_pred)
    SMAPE=smape(y, y_pred)
    return R2,MSE,RMSE,MAE,MAPE,SMAPE
def nfeatures_CV(df,y):
    global train_scores,cv_scores
    K_list=[]
    train_R2_list=[]
    train_RMSE_list=[]
    train_MSE_list=[]
    train_MAE_list=[]
    train_MAPE_list=[]
    train_SMAPE_list=[]
    cv_R2_list=[]
    cv_RMSE_list=[]
    cv_MSE_list=[]
    cv_MAE_list=[]
    cv_MAPE_list=[]
    cv_SMAPE_list=[]
    best_params_list=[]
    features_list=[]
    for i in range(1,28):
          model = SelectKBest(mutual_info_regression, k=i)
          model.fit_transform(X_unselect,y)
#print(X_new)
          var = model.get_support(True)
          X = X_unselect.iloc[:,var]
          features = X.columns.values
          features_list.append(features)
#    X = df
          K_list.append(i)
          print('K= %i' % i)
          model = RF(X,y)
#          model = GPR(X,y)
          train_R2,train_MSE,train_RMSE,train_MAE,train_MAPE,train_SMAPE=train_scores(model,X,y)
          cv_R2,cv_RMSE=cv_scores(model,X,y)
#    train_scores
          train_R2_list.append(train_R2)
          train_RMSE_list.append(train_RMSE)
          train_MSE_list.append(train_MSE)
          train_MAE_list.append(train_MAE)
          train_MAPE_list.append(train_MAPE)
          train_SMAPE_list.append(train_SMAPE)
          train_scores_list = {'K':K_list,
           'train_R2':train_R2_list,
         'train_RMSE':train_RMSE_list,
          'train_MSE':train_MSE_list,
          'train_MAE':train_MAE_list,
         'train_MAPE':train_MAPE_list,
        'train_SMAPE':train_SMAPE_list,
           'features':features_list
          }
          train_scores_list=pd.DataFrame(train_scores_list)
          train_scores_list.to_csv('./train_scores.csv',sep=',')
#cv_scores
          cv_R2_list.append(cv_R2)
          cv_RMSE_list.append(cv_RMSE)
          cv_scores_list = {'K':K_list,
           'cv_R2':cv_R2_list,
         'cv_RMSE':cv_RMSE_list,
           'features':features_list
          }
          cv_scores_list=pd.DataFrame(cv_scores_list)
          cv_scores_list.to_csv('./cv_scores.csv',sep=',')



      
if __name__ == '__main__':
      
    df=readdata('./df_afterpearson.csv')
#    features=readdata('./features_afterpearson.csv')
#    columns=features.columns.tolist()
    X_unselect =df.drop(['c/a','Unnamed: 0','Unnamed: 0.1'],axis=1)
    print(X_unselect.shape)
    y = df['c/a']
    nfeatures_CV(X_unselect,y)
