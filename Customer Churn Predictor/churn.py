import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create Mock Data
# In a real project, you would load a CSV using pd.read_csv('telco_churn.csv')
data = {
    'Monthly_Bill_USD': [50, 80, 100, 30, 120, 45, 90, 110, 40, 85],
    'Support_Calls': [1, 4, 5, 0, 6, 1, 3, 4, 0, 5],
    'Months_Subscribed': [24, 2, 1, 36, 1, 12, 4, 2, 48, 3],
    # Target: 1 means they Churned (Left), 0 means they Stayed
    'Churn': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1] 
}

df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[['Monthly_Bill_USD', 'Support_Calls', 'Months_Subscribed']]
y = df['Churn']

print("Dataset loaded! Here are the first few rows:")
print(df.head())

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the Model
# We are using a Random Forest for classification
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel has been trained!")

# Step 4: Make Predictions and Evaluate
predictions = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Step 5: Let's do a manual test!
print("\n--- Manual Prediction Test ---")
# Let's say we have a new customer: $95 bill, 4 support calls, subscribed for only 2 months
new_customer = pd.DataFrame([[95, 4, 2]], columns=['Monthly_Bill_USD', 'Support_Calls', 'Months_Subscribed'])
churn_prediction = model.predict(new_customer)

if churn_prediction[0] == 1:
    print("Prediction: This customer is HIGH RISK of churning! Send them a discount!")
else:
    print("Prediction: This customer is likely to stay.")