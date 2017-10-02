import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas_profiling
import seaborn as sns

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y = df_train['Loan_Status']

df_test_loanID = df_test['Loan_ID'].to_frame()

df_train.drop('Loan_ID', 1, inplace = True)
df_test.drop('Loan_ID', 1, inplace = True)

def data_explore(df_train):
    ser_gender = df_train['Gender'].value_counts()
    df_train['Gender'].fillna(ser_gender[ser_gender == max(ser_gender)].index[0], inplace = True)


    ser_married = df_train['Married'].value_counts()
    df_train['Married'].fillna(ser_married[ser_married == max(ser_married)].index[0], inplace = True)

    ser_dependents = df_train['Dependents'].value_counts()
    df_train['Dependents'].fillna(ser_dependents[ser_dependents == max(ser_dependents)].index[0], inplace = True)


    ser_self_employed = df_train['Self_Employed'].value_counts()
    df_train['Self_Employed'].fillna(ser_self_employed[ser_self_employed == max(ser_self_employed)].index[0], inplace = True)

    ser_loanamount = (df_train['LoanAmount'].median() + df_train['LoanAmount'].mean())/2
    df_train['LoanAmount'].fillna(ser_loanamount, inplace = True)

    ser_loanterm_amount = (df_train['Loan_Amount_Term'].median() + df_train['Loan_Amount_Term'].mean())/2
    df_train['Loan_Amount_Term'].fillna(ser_loanterm_amount, inplace = True)

    ser_credit_history = (df_train['Credit_History'].median() + df_train['Credit_History'].mean())/2
    df_train['Credit_History'].fillna(ser_credit_history, inplace = True)

    for column in df_train.columns:
        text_val_dict = {}
        if df_train[column].dtypes not in ('int64', 'float64'):
            column_contents = list(df_train[column].unique().astype(str))
            column_contents.sort()
            i = 0
            for content in column_contents:
                text_val_dict[content] = i
                i+=1
            df_train[column] = df_train[column].map(text_val_dict)
    return df_train

df_train_final = data_explore(df_train)
df_train_final.drop('Loan_Status', 1, inplace = True)
df_test_final = data_explore(df_test)


# Logistic Regression
model_log = LogisticRegression().fit(df_train_final, y)

df_test_final['Loan_Status'] = model_log.predict(df_test_final)
df_test_final = pd.concat([df_test_final, df_test_loanID], axis=1, join='inner')


df_test_final.to_csv('Final_submission.csv', columns = ['Loan_ID', 'Loan_Status'], index = False)