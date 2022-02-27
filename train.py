import pickle

import pandas as pd
import seaborn as sns
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from pandas_profiling import ProfileReport
from sklearn.metrics import roc_curve


from matplotlib import pyplot as plt 

# parameters
# We can pass the parameters on the command line so that we don't make
#changes to this file 
max_depth= 10
min_samples_leaf= 5 
n_estimators= 200
random_state= 1
output_file = f'model_Credit.bin'


# data preparation

df1 = pd.read_excel("Training Data Loan Cases.xlsx") #Data from our DB 
df2 = pd.read_excel("Training Customer Details.xlsx")#Data from CRB 

# join the two dataframes 
df = df1.merge(df2, how='left', on='CustomerId')

#drop Credit Score too many missing values
df.drop(['CreditScore'], axis=1, inplace=True)

#drop all missing rows 
df.dropna(inplace=True)

#Transforming objects to int 
#Transforming required numerical features to categorical
cols1 = ['PrequalifiedAmount','ScoreOutput_MobiLoansScore',
        'ScoreOutput_Probability','Account_FullSettledCount','MaximumMobileLoanPrincipalAmount_OtherSector',
       'AverageMobileLoanPrincipalAmount_MySector']
for i in cols1:
    df[i]= df[i].apply(int)

cols2 = ['ScoreOutput_Grade']
for c in cols2:
    lbl = LabelEncoder()
    lbl.fit(list(df[c].values))
    df[c] = lbl.transform(list(df[c].values))

cols3 = df.select_dtypes(include=['datetime64']).columns
cols3 = cols3.to_list()

cols4 = df.select_dtypes(include=['float64']).columns
cols4 = cols4.to_list()
for i in cols4:
    df[i]= df[i].apply(int)


#undersampling
defaulted = len(df[df["Status"] == 3])
paid_indices = df[df["Status"] == 6].index

random_indices = np.random.choice(paid_indices,defaulted,replace =False)
defaulted_indices = df[df["Status"] == 3].index

under_sample_indices = np.concatenate([defaulted_indices,random_indices])
under_sample = df.loc[under_sample_indices]


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

under_sample.drop(['Id','ReferenceNumber','CustomerId','ApprovedDate',
 'DisbursementDate',
 'StartDate',
 'NextPaymentDate',
 'CreatedDate',
 'NextExpectedPaymentDate',
 'EndDate',
 'DateOfBirth','CustomerCreatedDate','PersonalProfile_DateOfBirth'], axis=1,inplace=True )

df = under_sample

# validation
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.Status.values
y_val = df_val.Status.values



#training 
def train(df_train, y_train):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=1)
    model.fit(X_train, y_train)
    
    return dv, model

print("finished training")

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

print("finished predicting")

# Evaluating the model
dv, model = train(df_train_full, df_train_full.Status.values)

y_pred = predict(df_test, dv, model)

y_test = df_test.Status.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')