# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:45:49 2022

@author: Urmi
"""

import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import random
np.random.seed(20)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost
import shap


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#训练模型函数
#Train the Model
def model_train(Data_Train_X,Data_Train_Y,Data_Test_X,Data_Test_Y,Regress_Mode, data_apply_1, data_apply_2, data_apply_3, data_apply_4, data_apply_5, data_apply_6, data_apply_7, data_apply_8, data_apply_9, data_apply_10, data_apply_11, data_apply_12, data_apply_13, data_apply_14, data_apply_15, data_apply_16, data_apply_17):
    #Polynomial_Regression
    if Regress_Mode == 0:
        polynomial_features= PolynomialFeatures(degree = 2)
        Data_Train_X = polynomial_features.fit_transform(Data_Train_X)
        
        polynomial_features= PolynomialFeatures(degree=2)
        Data_Test_X = polynomial_features.fit_transform(Data_Test_X)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_1 = polynomial_features.fit_transform(data_apply_1)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_2 = polynomial_features.fit_transform(data_apply_2)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_3 = polynomial_features.fit_transform(data_apply_3)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_4 = polynomial_features.fit_transform(data_apply_4)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_5 = polynomial_features.fit_transform(data_apply_5)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_6 = polynomial_features.fit_transform(data_apply_6)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_7 = polynomial_features.fit_transform(data_apply_7)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_8 = polynomial_features.fit_transform(data_apply_8)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_9 = polynomial_features.fit_transform(data_apply_9)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_10 = polynomial_features.fit_transform(data_apply_10)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_11 = polynomial_features.fit_transform(data_apply_11)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_12 = polynomial_features.fit_transform(data_apply_12)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_13 = polynomial_features.fit_transform(data_apply_13)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_14 = polynomial_features.fit_transform(data_apply_14)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_15 = polynomial_features.fit_transform(data_apply_15)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_16 = polynomial_features.fit_transform(data_apply_16)
        
        polynomial_features= PolynomialFeatures(degree=2)
        data_apply_17 = polynomial_features.fit_transform(data_apply_17)
        

                
        model = LinearRegression()

    #Extra Tree
    elif Regress_Mode == 1 :
        model = ExtraTreesRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                            max_depth=200, max_features='auto', max_leaf_nodes=None,
                            max_samples=None, min_impurity_decrease=0.0,
                            min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=15, n_jobs=None, oob_score=False,
                            random_state=42, verbose=0, warm_start=False)
    elif Regress_Mode == 2 :
    #Random Forest
        model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                max_depth=500, max_features=None, max_leaf_nodes=80,
                                max_samples=None, min_impurity_decrease=0.0,
                                min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                n_estimators=300, n_jobs=-1, oob_score=False,
                                random_state=20, verbose=0, warm_start=False) #min_impurity_split=None,
    elif Regress_Mode == 3 :
            #XGBoost
        model = xgboost.XGBRegressor(bootstrap=True, max_depth=300, max_features=None)

    model.fit(Data_Train_X,Data_Train_Y)
    pred_y = model.predict(Data_Train_X) #Train_Predict
    pred_yy = model.predict(Data_Test_X) #Test_Predict
    
    # if If_Predict_Data == 1:
    pred_1 = model.predict(data_apply_1)
    pred_2 = model.predict(data_apply_2)
    pred_3 = model.predict(data_apply_3)
    pred_4 = model.predict(data_apply_4)
    pred_5 = model.predict(data_apply_5)
    pred_6 = model.predict(data_apply_6)
    pred_7 = model.predict(data_apply_7)
    pred_8 = model.predict(data_apply_8)
    pred_9 = model.predict(data_apply_9)
    pred_10 = model.predict(data_apply_10)
    pred_11 = model.predict(data_apply_11)
    pred_12 = model.predict(data_apply_12)
    pred_13 = model.predict(data_apply_13)
    pred_14 = model.predict(data_apply_14)
    pred_15 = model.predict(data_apply_15)
    pred_16 = model.predict(data_apply_16)
    pred_17 = model.predict(data_apply_17)

     
    np.savetxt('./pred_1.txt',pred_1)
    np.savetxt('./pred_2.txt',pred_2)
    np.savetxt('./pred_3.txt',pred_3)
    np.savetxt('./pred_4.txt',pred_4)
    np.savetxt('./pred_5.txt',pred_5)
    np.savetxt('./pred_6.txt',pred_6)
    np.savetxt('./pred_7.txt',pred_7)
    np.savetxt('./pred_8.txt',pred_8)
    np.savetxt('./pred_9.txt',pred_9)
    np.savetxt('./pred_10.txt',pred_10)
    np.savetxt('./pred_11.txt',pred_11)
    np.savetxt('./pred_12.txt',pred_12)
    np.savetxt('./pred_13.txt',pred_13)
    np.savetxt('./pred_14.txt',pred_14)
    np.savetxt('./pred_15.txt',pred_15)
    np.savetxt('./pred_16.txt',pred_16)
    np.savetxt('./pred_17.txt',pred_17)

    print('Slope:' , model.coef_)
    print('Intercept:', model.intercept_)
    pd.DataFrame(pred_y).to_csv("file3.csv")
    pd.DataFrame(pred_yy).to_csv("file4.csv")
    #impute the error parameter
    #R2
    r2_model = sm.r2_score(Data_Train_Y, pred_y) #R2 impute TrainData's R2
    r2_test = sm.r2_score(Data_Test_Y, pred_yy)  #impute TestData's R2

    print(r2_model)


    
#RMSE
    rmse_model = np.sqrt(sm.mean_squared_error(Data_Train_Y, pred_y)) #RMSE impute TrainData's RMESE
    rmse_test = np.sqrt(sm.mean_squared_error(Data_Test_Y, pred_yy)) #RMSE impute TestData's RMSE

     #Keep three decimals
    r2_model = round(r2_model,3)
    r2_test = round(r2_test,3)
    rmse_model = round(rmse_model,3)
    rmse_test = round(rmse_test,3)


    #tempr = Error Parameter Set
    temp_r =[]
    temp_r.append(r2_model)
    temp_r.append(r2_test)
    temp_r.append(rmse_model)
    temp_r.append(rmse_test)


    # plot
    binary_plot(y_train = Data_Train_Y,
                y_train_label = pred_y,
                y_test = Data_Test_Y,
                y_test_label = pred_yy,
                train_rmse = rmse_model,
                test_rmse = rmse_test,
                train_r2 = r2_model,
                test_r2 = r2_test)
    save_fig("Result_plot")
    
    
    #save the result 




# make plot by sany He
def binary_plot(y_train,  y_train_label, y_test, y_test_label,
                train_rmse, test_rmse, train_r2, test_r2,
                text_position=[1000.5, 2000.075]):
    """plot the binary diagram

    :param y_train: the label of the training data set
    :param y_train_label: the prediction of the training the data set
    :param y_test: the label of the testing data set
    :param y_test_label: the prediction of the testing data set
    :param train_rmse: the RMSE score of the training data set
    :param test_rmse: the RMSE score of the testing data set
    :param train_r2: the R2 score of the training data set
    :param test_r2: the R2 score of the testing data set
    :param test_position: the coordinates of R2 text for
    """
    
    plt.figure(figsize=(6,6))
    plt.scatter(y_train, y_train_label, marker="s",
                label="Training set-RMSE={}".format(train_rmse))
    plt.scatter(y_test, y_test_label, marker="o",
                label="Test set-RMSE={}".format(test_rmse))
    plt.legend(loc="upper left", fontsize=14)
    plt.xlabel("Reference value", fontsize=20)
    plt.ylabel("Predicted value", fontsize=20)
    a=[0,10000]; b=[0,10000]
    plt.plot(a, b)
    plt.text(text_position[0]+100.5, text_position[0]+6000.0,
             r'$R^2(train)=${}'.format(train_r2),
             fontdict={'size': 16, 'color': '#000000'})
    plt.text(text_position[0]+100.5, text_position[0]+5000.5,
             r'$R^2(test)=${}'.format(test_r2),
             fontdict={'size': 16, 'color': '#000000'})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim((0, 10000))


#save figure 
def save_fig(fig_id, tight_layout=True):
    '''
    Run to save automatic pictures
    
    :param fig_id: image name
    '''
    path = "./Image/"+fig_id+".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)



#input data：data_ is source_data，data_0 is "label = 0" data，data_1 is "label = 1"data,data_2 is"label = 2"data
import pandas as pd
#data_ = pd.read_csv("run_new.csv", low_memory=False)

data_0 = np.loadtxt('./dataset/Data_0.txt')
data_1 = np.loadtxt('./dataset/Data_1.txt')
data_2 = np.loadtxt('./dataset/Data_2.txt')
data_3 = np.loadtxt('./dataset/Data_3.txt')
data_4 = np.loadtxt('./dataset/Data_4.txt')
data_5 = np.loadtxt('./dataset/Data_5.txt')
data_6 = np.loadtxt('./dataset/Data_6.txt')
data_7 = np.loadtxt('./dataset/Data_7.txt')


data_apply_1 = np.loadtxt('./Mars3GPa.txt')
data_apply_2 = np.loadtxt('./Pred_Earth_Bulk_HT.txt')
data_apply_3 = np.loadtxt('./Pred_Earth_Komatiite_IT.txt')
data_apply_4 = np.loadtxt('./Pred_Earth_MORB_LT.txt')
data_apply_5 = np.loadtxt('./Pred_Mars_basalt_Dry_HT.txt')
data_apply_6 = np.loadtxt('./Pred_Mars_basalt_Dry_IT.txt')
data_apply_7 = np.loadtxt('./Pred_Mars_basalt_Dry_LT.txt')
data_apply_8 = np.loadtxt('./Pred_Mars_basalt_Wet_HT.txt')
data_apply_9 = np.loadtxt('./Pred_Mars_basalt_Wet_IT.txt')
data_apply_10 = np.loadtxt('./Pred_Mars_basalt_Wet_LT.txt')
data_apply_11 = np.loadtxt('./Pred_Moon_Longhi_HT.txt')
data_apply_12 = np.loadtxt('./Pred_Moon_Longhi_IT.txt')
data_apply_13 = np.loadtxt('./predict_dataset/Pred_Moon_Longhi_LT.txt')
data_apply_14 = np.loadtxt('./predict_dataset/Pred_Moon_Taylor_HT.txt')
data_apply_15 = np.loadtxt('./predict_dataset/prediction/Pred_Moon_Taylor_IT.txt')
data_apply_16 = np.loadtxt('./predict_dataset/Pred_Moon_Taylor_LT.txt')
data_apply_17 = np.loadtxt('./predict_dataset/Steenstra.txt')



#Disarrange the sample order of the data set
np.random.shuffle(data_0)
np.random.shuffle(data_1)
np.random.shuffle(data_2)
np.random.shuffle(data_3)
np.random.shuffle(data_4)
np.random.shuffle(data_5)
np.random.shuffle(data_6)
np.random.shuffle(data_7)


#DataMode：Stratified sampling was conducted directly (7:3)
x1 = data_0[0:int(data_0.shape[0]*0.7)] 
x2 = data_1[0:int(data_1.shape[0]*0.7)] 
x3 = data_2[0:int(data_2.shape[0]*0.7)]
x4 = data_3[0:int(data_3.shape[0]*0.7)]
x5 = data_4[0:int(data_4.shape[0]*0.7)]
x6 = data_5[0:int(data_5.shape[0]*0.7)]
x7 = data_6[0:int(data_6.shape[0]*0.7)]
x8 = data_7[0:int(data_7.shape[0]*0.7)]
Data_Train = np.vstack((x1,x2,x3,x4,x5,x6,x7,x8))
print(Data_Train)


x1 = data_0[int(data_0.shape[0]*0.7):int(data_0.shape[0])] 
x2 = data_1[int(data_1.shape[0]*0.7):int(data_1.shape[0])] 
x3 = data_2[int(data_2.shape[0]*0.7):int(data_2.shape[0])]
x4 = data_3[int(data_3.shape[0]*0.7):int(data_3.shape[0])]
x5 = data_4[int(data_4.shape[0]*0.7):int(data_4.shape[0])]
x6 = data_5[int(data_5.shape[0]*0.7):int(data_5.shape[0])]
x7 = data_6[int(data_6.shape[0]*0.7):int(data_6.shape[0])]
x8 = data_7[int(data_7.shape[0]*0.7):int(data_7.shape[0])]
Data_Test = np.vstack((x1,x2,x3,x4,x5,x6,x7,x8))
print(Data_Test)

pd.DataFrame(Data_Test).to_csv("file1.csv")
pd.DataFrame(Data_Train).to_csv("file2.csv")

Data_Train_X = Data_Train[:,0:-1]
Data_Train_Y = Data_Train[:,-1]
print(Data_Train_Y)
Data_Test_X = Data_Test[:,0:-1]
Data_Test_Y = Data_Test[:,-1]
print(Data_Test_X)
print(Data_Test_Y)
Regress_Mode = 3  #0-linear 1-extratree 2-Random forest 3-XGBoost
list_R2_RMSE = model_train(Data_Train_X, Data_Train_Y, Data_Test_X, Data_Test_Y,Regress_Mode, data_apply_1, data_apply_2, data_apply_3, data_apply_4, data_apply_5, data_apply_6, data_apply_7, data_apply_8, data_apply_9, data_apply_10, data_apply_11, data_apply_12, data_apply_13, data_apply_14, data_apply_15, data_apply_16, data_apply_17)






