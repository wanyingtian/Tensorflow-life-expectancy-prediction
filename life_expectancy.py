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