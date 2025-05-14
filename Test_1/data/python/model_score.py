import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("processed_data.csv")
X = df[["CK", "QT_x", "QT_y"]]  # Features (Predictors)
y = df["HP_grade"]  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Visualize actual vs predicted
plt.scatter(y_test, y_pred, color="crimson", alpha=0.6, edgecolors="black")
plt.xlabel("Actual HP_Grade")
plt.ylabel("Predicted HP_Grade")
plt.title("Actual vs Predicted HP_Grade")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

X = df[["CK", "QT_x", "QT_y", "KT_TB", "KT1"]]  # Features (Predictors)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Visualize actual vs predicted
plt.scatter(y_test, y_pred, color="crimson", alpha=0.6, edgecolors="black")
plt.xlabel("Actual HP_Grade")
plt.ylabel("Predicted HP_Grade")
plt.title("Actual vs Predicted HP_Grade")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
