import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

## Data Loading and Observing
# Load the data csv file to dataset
dataset = pd.read_csv('Life_Expectancy_Data.csv')
# print the dataset to visualize
print(dataset.head())
print(dataset.describe())
# Drop the country and  continent columns as they shouldn't influence life expectancy 
dataset = dataset.drop(['Country'], axis = 1)
dataset = dataset.drop(['Continent'], axis = 1)
#dataset = dataset.drop(['Status'], axis = 1)
# # split data into label and features
label_index=dataset.columns.get_loc("Life_expectancy ")
labels = dataset.iloc[:,label_index] 
features = dataset.drop(["Life_expectancy "], axis = 1)

## Data Preprocessing
# apply one-hot-encoding on all categorial columns
features = pd.get_dummies(features)
# split data into training and test sets
features_train,features_test,labels_train,labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)
#standardize the numeric columns using ColumnTransformer
numerical_features = features.select_dtypes(include=['float64','int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.fit_transform(features_test)
print('done')