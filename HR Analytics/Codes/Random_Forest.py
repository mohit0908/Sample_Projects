import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split



df = pd.read_excel('HR_comma_sep.xlsx')

## Create additional variables from existing variables
# Convert weekly data to monthly data
df['average_weekly_hours'] = df['average_montly_hours']/4

# Converting yearly data to monthly data
df['time_spend_company_months'] = df['time_spend_company']*12

# Convert salary categorical variable to numerica; variabls
df['salary'] = df['salary'].map({'low':1, 'medium': 2, 'high': 3})\

# Encoding sales data.
le = LabelEncoder()
df['sales_encoded'] = le.fit_transform(df['sales'])
cols = [x for x in df.dtypes.index if df.dtypes[x] != 'object']
X = df[cols].drop('left', axis = 1)
y = df['left']

# Dividing data into training and testing data sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =0)
RDF_Model = RandomForestClassifier().fit(X_train, y_train)
print('Test score: ', RDF_Model.score(X_test, y_test))