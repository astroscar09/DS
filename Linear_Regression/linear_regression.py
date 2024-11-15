import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual Prices vs Predicted Prices')
    plt.show()

# Load the dataset
print('Loading the Housing Dataset')
data = pd.read_csv('/Users/oac466/Desktop/DS/Linear_Regression/Housing.csv')

# The target variable is 'price' and the rest are features
X = data.drop('price', axis=1)

#to bring the mse down and make the model more accurate, we can take the log of the price
#we do this because the price is too high 
#and the model will not be able to predict it accurately
y = np.log10(data['price'])

# Convert binary columns to numeric
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

for col in binary_columns:
    X[col] = X[col].map({'yes': 1, 'no': 0})


X['furnishingstatus'] = X['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})   

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Generating the linear regression model')
# Create a linear regression model
model = LinearRegression()

# Train the model
print('Training the model...')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

score = model.score(X_test, y_test)
print("Model R^2 Score:", score)


plot_predictions(y_test, y_pred)