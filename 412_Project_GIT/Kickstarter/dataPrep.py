import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn
import keras
import tensorflow
train = pd.read_csv('Data/trainCleanest.csv')
test = pd.read_csv('Data/testCleanest.csv')
alld = pd.concat([train, test], axis = 0)
y_train = train['state']
y_test = test['state']

columns_to_encode = ['category', 'main_category', 'currency', ]
columns_to_scale = ['goal', 'pledged', 'backers', 'usd.pledged', 'usd_pledged_real', 'usd_goal_real', 'period']
columns_to_scale = ['goal', 'backers', 'period']

# Instantiate encoder/scaler
ohe = OneHotEncoder(sparse=False)

###print(sum(alld[columns_to_encode].isnull()))

ohe.fit(alld[columns_to_encode])

# Scale and Encode Separate Columns
scaled_columns = train[columns_to_scale]
encoded_columns = ohe.transform(train[columns_to_encode])

train = np.concatenate([scaled_columns, encoded_columns], axis=1)

scaled_columns = test[columns_to_scale]
encoded_columns = ohe.transform(test[columns_to_encode])

test = np.concatenate([scaled_columns, encoded_columns], axis=1)


