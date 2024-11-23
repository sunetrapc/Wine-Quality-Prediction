# Importing necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Setting random seeds for reproducibility
random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)

# Loading the dataset
data = pd.read_excel("wine quality with category value.xlsx")

# Exploratory Data Analysis (EDA)
print("First few rows of the data:")
print(data.head())

print("Summary of the data:")
print(data.info())

print("Statistics of the dataset:")
print(data.describe())

# Visualizing data distribution
sns.histplot(data['quality'], kde=False, bins=10)
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Input (x) and target (y) data
x = data.drop(columns=['quality'])  # Input data excluding quality
y = data['quality']  # Target is the "quality" column

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1693)

# Creating a sequential deep learning model
model = Sequential(name="WineQuality")

# Add layers to the model
model.add(Dense(7, input_dim=x_train.shape[1], activation='relu', name="HL1"))  # First 
model.add(Dense(5, activation='relu', name="HL2"))  # Second 
model.add(Dense(1, activation='linear', name="OL"))  # Output 

# Displaying model summary
model.summary()

# Compiling the model
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Training the model (150 epochs, 25% validation split, no output during training)
training_history = model.fit(x_train, y_train, validation_split=0.25, epochs=150, verbose=0)

# Evaluating the model
print("Predicted values for the first two records:")
predicted_values = model.predict(x.iloc[:2])
print(predicted_values)

print("Model MSE value from the last training epoch:", training_history.history['val_mse'][-1])

# Evaluating Precision, Recall, F1-score (Assuming classification context; adapt for regression if needed)
y_test_pred = model.predict(x_test).flatten()
y_test_pred = np.round(y_test_pred)  # Convert predictions to integers for classification metrics
print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
print("F1-Score:", f1_score(y_test, y_test_pred, average='weighted'))

