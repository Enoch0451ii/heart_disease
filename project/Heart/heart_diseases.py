# Load the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('project\Heart\heart_disease_data.csv')

# Preprocess the data
X = data.drop(columns='target', axis=1)
Y = data['target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create the model
model = LogisticRegression(max_iter=1000)

# Training the logisticRegression with training data
model.fit(x_train, y_train)

# Model evaluation
# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy on training data : ', training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy on test data : ', test_data_accuracy)

# Building a predictive system
input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
    print('The person does not have a heart disease')
else:
    print('The person has a heart disease')

# Save the trained model as a .pkl file
filename = 'heart_disease_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved successfully as heart_disease_model.pkl")
