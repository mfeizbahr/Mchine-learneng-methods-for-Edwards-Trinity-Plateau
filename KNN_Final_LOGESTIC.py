# Import required libraries
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv("C:/pr2/a_with_y_pred_probwa.csv")

# Preprocess the data
X = data[["LatitudeDD_x", "LongitudeDD_x"]]
y = data["y_pred_probwa"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the KNN regressor model
knn = KNeighborsRegressor(n_neighbors=5)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the values using the test data
y_pred = knn.predict(X_test)

# Predict all valuse
y_pred_all = knn.predict(X)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print("Mean Squared Error:", mse)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print("R-squared:", r2)