import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load earthquake data (You will need a dataset containing earthquake records)
# Here, we assume you have a dataset with columns: latitude, longitude, magnitude, depth, etc.
earthquake_data = pd.read_csv("earthquake_data.csv")

# Define a target variable for earthquake risk (e.g., 1 for high risk, 0 for low risk)
earthquake_data['risk'] = np.where(earthquake_data['magnitude'] >= 5.0, 1, 0)

# Select relevant features (e.g., latitude, longitude, depth, etc.)
X = earthquake_data[['latitude', 'longitude', 'depth']]

# Define the target variable
y = earthquake_data['risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier (you can choose a different classifier if desired)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# You can now use this model to predict earthquake risk for new data points.
