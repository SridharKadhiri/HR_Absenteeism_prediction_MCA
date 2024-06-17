#!/usr/bin/env python
# coding: utf-8

# #### Absenteeism Module testing file 
# 
#     creating a module with methods
#     
#         1. load_clean_data  - loads the new data and cleans it along with scaling
#         2. predicted_probability - gets the probability for each record
#         3. predicted_output_category - gets the predited value 
#         4. predicted_outputs - displays the dataframe with probability and prediction
#         5. __init__

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import pickle
from  sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler():
    def __init__ (self,columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scale_std = StandardScaler()
        
    def fit_transform(self,df):
        scaled_data = df.copy()
        scaled_data[self.columns_to_scale] = self.scale_std.fit_transform(df[self.columns_to_scale])
        return scaled_data

class Absenteeism_model():
    """
    1. load_n_clean  - loads the new data and cleans it along with scaling
    
    2. predicted_probability - gets the probability for each record
    
    3. predicted_output_category - gets the predited value 
    
    4. predicted_outputs - displays the dataframe with probability and prediction
    
    5. __init__
    
    """
    def __init__(self):
        with open('Model','rb') as model_file , open('Scaler','rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
#             self.data = None
    
    def load_n_clean(self,data_file):
        # read the file in the argument
        df = pd.read_excel(data_file) 
#         print(f'Read the file {data_file} : Done')
        
        df.drop(['ID'],axis =1 ,inplace = True ) 
#         print(f'Dropped the ID column : Done')
        
        required_columns = ['Reason for absence',
                    'Month of absence',
                    'Day of the week', 
                    'Transportation expense',
                    'Distance from Residence to Work',
                    'Work load Average/day ',
                    'Education',
                    'Son',
                    'Pet',
                    'Body mass index',
                    'Absenteeism time in hours']
        
        df = df[required_columns]
#         print('Removing the unnecessay columns : Done')
        
        df.dropna(inplace = True)
#         print('Dropping the null values : Done')
        
        df = df.astype(int)
#         print('converting the data type to int : Done' )

        df['Excessive Absentise'] = np.where(df['Absenteeism time in hours'] >\
                                             df['Absenteeism time in hours'].median(),1,0)
               
        df.drop(['Absenteeism time in hours'],inplace = True, axis = 1)
#         print("dropping the df['Absenteeism time in hours'] column")
        reason_dummies = pd.get_dummies(df['Reason for absence'],drop_first=True)
        
        reason_1_g = reason_dummies.loc[:,1:14].max(axis =1)
        reason_2_g = reason_dummies.loc[:,15:17].max(axis =1)
        reason_3_g = reason_dummies.loc[:,18:21].max(axis =1)
        reason_4_g = reason_dummies.loc[:,21:].max(axis =1)
        
        df = pd.concat([df,reason_1_g,reason_2_g,reason_3_g,reason_4_g],axis = 1)
        
        column_names = ['Reason for absence', 'Month of absence', 'Day of the week',
                       'Transportation expense', 'Distance from Residence to Work',
                       'Work load Average/day ', 'Education', 'Son', 'Pet','Body mass index',
                        'Excessive Absentise', 'Reason_1','Reason_2','Reason_3','Reason_4']
        df.columns  = column_names
        
        reorder_col = ['Reason_1','Reason_2','Reason_3','Reason_4','Reason for absence',
                       'Month of absence', 'Day of the week','Transportation expense',
                       'Distance from Residence to Work','Work load Average/day ',
                       'Education', 'Son', 'Pet','Body mass index', 'Excessive Absentise']
        
        df = df[reorder_col]
        
        df.drop(['Reason for absence'],axis = 1,inplace = True)
        
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        df.to_csv('processed_data_.csv')
        
        self.data = df.copy()
        
#         print('Returning the 1st five rows')
#         return df.head()
        
    def predicted_probability(self,df):
        pred = self.reg.predict_proba(df)[:,-1]
        return pred

    def predicted_output_category(self,df):
        predict = self.reg.predict(df)
        return predict
        
    def predicted_outputs(self,df):
        pred = self.reg.predict_proba(df)[:,-1]
        predict = self.reg.predict(df)
        self.data["Probability"] = pred
        self.data['Prediction'] = predict
        return self.data
            
            


# # ### Data Cleaning

# # In[ ]:


# get_ipython().run_cell_magic('time', '', "data = 'data_raw.xls'\nab_mod = Absenteeism_module()\n\nab_mod.load_n_clean(data)\n\nclean_data = ab_mod.data\nclean_data.head()\n")


# # In[ ]:





# # In[ ]:


# columns_to_scale = ['Month of absence',
#        'Day of the week', 'Transportation expense',
#        'Distance from Residence to Work', 'Work load Average/day ','Son', 'Pet', 'Body mass index',
#        'Excessive Absentise']

# scaler = CustomScaler(columns_to_scale)


# # In[ ]:


# scaled_data = scaler.fit_transform(clean_data)
# scaled_data.head()


# # In[ ]:


# ab_mod.predicted_outputs(scaled_data.iloc[:,:-1])


# # In[ ]:




