# Step 1: Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Load the Data
# We are using a built-in dataset of California housing prices
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names) # Features (House Age, Rooms, etc.)
y = housing.target # Target (The Price we want to predict)

print("Data loaded! Here are the features we are looking at:")
print(X.head()) 

# Step 3: Train/Test Split
# We split the data: 80% for training the model, 20% for testing it.
# random_state ensures we get the same split every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose and Train the Model
# We are using Linear Regression: it draws a line of best fit through the data points
model = LinearRegression()

# 'fit' is the training step. The model is learning the relationship between X (features) and y (price)
model.fit(X_train, y_train)
print("\nModel has been trained!")

# Step 5: Make Predictions and Evaluate
# Now we ask the model to predict prices for the 20% test data it hasn't seen
predictions = model.predict(X_test)

# We check how far off our predictions are from the actual prices (y_test)
mae = mean_absolute_error(y_test, predictions)
print(f"\nMean Absolute Error: {mae:.2f}")
print("This means our model's predictions are off by an average of {:.2f} (in hundreds of thousands of dollars)".format(mae))

# Step 6: Let's do a manual test!
# Showing the actual price vs what our model predicted for the first test house
print(f"\nActual Price of first test house: {y_test[0]:.2f}")
print(f"Our Model's Prediction: {predictions[0]:.2f}")