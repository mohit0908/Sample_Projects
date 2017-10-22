from scipy.stats import mode
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train, test],ignore_index=True)

cat_col = [x for x in data.dtypes.index if data.dtypes[x] == 'object']
cat_cols = [x for x in cat_col if x not in ['Item_Identifier', 'Outlet_Identifier']]

## Standardize Item_Fat_Content Column

data['Item_Fat_Content'] = data['Item_Fat_Content'].apply(lambda x: 'Low Fat' if x in ['Low Fat', 'LF', 'low fat'] 
                               else 'Regular' if x in ['Regular', 'reg'] else x) 


## Impute Item_Weight column values

item_avg_weight = data.pivot_table(index = 'Item_Identifier', aggfunc='mean', values = 'Item_Weight')
miss_bol = data.Item_Weight.isnull()
data.loc[miss_bol, 'Item_Weight'] = data.loc[miss_bol, 'Item_Identifier'].apply(lambda x: item_avg_weight[x])

## Impute Outlet_Size column values

outlet_size_mode = data.dropna().pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
miss_bool = data['Outlet_Size'].isnull()
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

## Imputing 0 Item_Visibility values with mean of Item_Visibility for that Item Identifier

Item_Viz = data.pivot_table(aggfunc = 'mean', index = 'Item_Identifier', values = 'Item_Visibility')
miss_bool = (data['Item_Visibility'] == 0)
data.loc[miss_bool, 'Item_Visibility'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: Item_Viz[x])

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food', 'DR':'Drinks', 'NC': 'Non-Consumerable'})
# data.pivot_table(index = 'Item_Type', values = ['Item_Outlet_Sales'], 
#                   aggfunc = [np.max, np.mean])
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data.loc[data['Item_Type_Combined'] == 'Non-Consumerable','Item_Fat_Content' ] = 'Non-Edible'

le = LabelEncoder()

data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])


le = LabelEncoder()
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type']

for i in var_mod:
    data[i] = le.fit_transform(data[i])
    

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])

data.drop(['Item_Type','Outlet_Establishment_Year' ], axis = 1, inplace = True)

train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']

train.drop(['source'], axis = 1, inplace = True)
test.drop(['source', 'Item_Outlet_Sales'], axis = 1, inplace = True)

train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


def model_func(algo, train_f, test_f, predict_cols, result_cols, target, filename):
    model = algo.fit(train_f[predict_cols], train_f[target])
    
    train_predict = algo.predict(train_f[predict_cols])
    
    cv_score = cross_validation.cross_val_score(algo, train_f[predict_cols], train_f[target], 
                                                cv = 20, scoring = 'neg_mean_squared_error')
#     cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    print('Model report')
    print('RMSE: ', np.sqrt(metrics.mean_squared_error(train_f[target].values, train_predict )))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),
                                                                              np.std(cv_score),np.min(cv_score),
                                                                              np.max(cv_score)))
    
    # Predicting on test data
    
    test_f[target] = algo.predict(test_f[predict_cols])
    
    # Exporting submission file
    
    result_cols.append(target)
    submission = pd.DataFrame({ x: test_f[x] for x in result_cols})
    submission.to_csv(filename, index = False)
    return

# Random Forest Model

target = 'Item_Outlet_Sales'
result_cols = ['Item_Identifier','Outlet_Identifier']

predictors = [x for x in train.columns if x not in [target] + result_cols]

RDT_Model = RandomForestRegressor(n_estimators=100,max_depth=10, min_samples_leaf=100,n_jobs=4)
model_func(RDT_Model, train, test, predictors, result_cols, target, 'Random_Forest2.csv')


