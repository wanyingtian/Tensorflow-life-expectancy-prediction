import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.optimizers import Adam

## Data Loading and Observing
# Load the data csv file to dataset
dataset = pd.read_csv('Life_Expectancy_Data.csv')
dataset = dataset.dropna()
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
print("Data preprocessing...")
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

## Building the model
print("Building the model...")
# create an instance of sequential model
my_model = Sequential()
# create input layer, shape is number of features in dataset
input = InputLayer(input_shape = (features.shape[1],))
# add input layer to my model
my_model.add(input)
# add hidden layer with relu activation function, 64 hidden units
# WHY 64
my_model.add(Dense(64, activation = 'relu'))
# add output layer - single output regression, 1 neuron
my_model.add(Dense(1))
# print model summary
print(my_model.summary())

## Initializing Optimizer, and compiling model
print("Initializing Optimizer, and compiling model..")
# Create an instance of the Adam optimizer with the learning rate equal to 0.01.
opt = Adam(learning_rate = 0.001)
# Compile: loss use the Mean Squared Error (mse);  metrics use the Mean Absolute Error (mae)
my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

## Fit and Evaluate the model
print("Fitting the model...")
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size = 1, verbose = 1)

print("Evaluating the model...")
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)
print("Final loss is:" + str(res_mse) + ", final metric is:" + str(res_mae))
print('done')
