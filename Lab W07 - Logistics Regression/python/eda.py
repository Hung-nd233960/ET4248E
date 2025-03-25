import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Lab W07 - Logistics Regression/data/data.csv")

# Display the number of rows and columns
print(df.shape)
print(df.head())
print(df.info())  # Display data type of each column
print(df.describe())  # Display statistical information of numerical columns
print(df.duplicated().sum())  # Check for duplicates
print(df.isnull().sum())  # Check for missing values

# 5 rows and 16 columns
# 0 duplicates
# 0 missing values

# Columns that are having inappropriate data:
# - patient_number: This column is not useful for analysis
# - gender: This column need label encoding
# - chol_hdl_ratio, bmi, waist_hip_ratio: This column need to be converted from int to float
# - diabetes; This column need label encoding

# Step 1: Drop the 'patient_number' column
df.drop(columns=["patient_number"], inplace=True)

# Step 2: Label encoding for 'gender' and 'diabetes'
df["gender"] = df["gender"].map({"male": 0, "female": 1})
df["diabetes"] = df["diabetes"].map({"No diabetes": 0, "Diabetes": 1})

# Step 3: Convert specified columns to float
columns_to_float = ["chol_hdl_ratio", "bmi", "waist_hip_ratio"]
for col in columns_to_float:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# Display the number of rows and columns
print(df.shape)
print(df.head())
print(df.info())  # Display data type of each column
print(df.describe())  # Display statistical information of numerical columns
print(df.duplicated().sum())  # Check for duplicates
print(df.isnull().sum())  # Check for missing values

# Create a histogram for each numerical column
df.hist(bins=20, figsize=(15, 10), color="skyblue", edgecolor="black")
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True
)
plt.title("Correlation Heatmap of Dataset")
plt.show()

# Corellation shows that 'glucose', 'age', 'chol_hdl_ratio' and 'diabetes' are highly correlated
# Let's select these features for further analysis
selected_features = [
    "glucose",
    "age",
    "chol_hdl_ratio",
    "diabetes",
]  # Assuming 'diabetes' is the target variable
df_filtered = df[selected_features]

# Choose model: Logistic Regression

# Save the filtered dataset
df_filtered.to_csv(
    "Lab W07 - Logistics Regression/data/filtered_dataset.csv", index=False
)

print("Filtered dataset saved as 'filtered_dataset.csv")
