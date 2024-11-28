import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('/Users/oac466/Desktop/DS/Linear_Regression/co2.csv')

# Display the first few rows of the dataset
print(data.head())

# Define the features (X) and the target (y)
# Assuming 'CO2' is the target variable and the rest are features
X = data.drop('CO2', axis=1)
y = data['CO2']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Display the coefficients
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)