import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

df = pd.read_excel("dataset.xlsx", sheet_name="Sheet1")  # Load Excel file
print(df.head())  # Display first 5 rows
df.drop(columns=["MSSV", "Họ và tên SV"], inplace=True)  # Delete specific column
df.to_csv("output.csv", index=False)  # Save as CSV without index
df.to_excel("output.xlsx", index=False, sheet_name="Processed")  # Save as Excel

df = pd.read_csv("output.csv")  # Load CSV file

# Display the number of rows and columns
print(df.shape)
print(df.head())
print(df.info())  # Display data type of each column
print(df.describe())  # Display statistical information of numerical columns
print(df.isnull().sum())  # Check for missing values
print(df.duplicated().sum())  # Check for duplicates
df.fillna(df.mean(), inplace=True)
print(df.isnull().sum())

plt.hist(df["HP_grade"], bins=20, color="blue", edgecolor="black", alpha=0.7)

# Formatting
plt.xlabel("HP Grade")
plt.ylabel("Frequency")
plt.title("Histogram of HP Grade")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()

sns.boxplot(x=df["HP_grade"])
plt.show()

df = df[df["HP_grade"] >= 3]

plt.hist(df["HP_grade"], bins=20, color="blue", edgecolor="black", alpha=0.7)

# Formatting
plt.xlabel("HP Grade")
plt.ylabel("Frequency")
plt.title("Histogram of HP Grade")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()

# Create subplots for all numeric columns
df.select_dtypes(include=["number"]).hist(
    bins=20, figsize=(10, 8), color="blue", edgecolor="black"
)

# Ensure y-axis is whole numbers
for ax in plt.gcf().axes:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.suptitle("Histograms of All Numeric Columns", fontsize=14)
plt.show()

corr_matrix = df.corr()
# Set figure size
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add title
plt.title("Correlation Matrix Heatmap")

# Show plot
plt.show()

# Loop through numeric columns and create individual boxplots
df.select_dtypes(include=["number"]).plot(
    kind="box",
    subplots=True,
    layout=(2, 5),
    figsize=(12, 8),
    sharex=False,
    sharey=False,
)

plt.suptitle("Boxplots of Each Numeric Column")
plt.show()

df = df[df["QT_x"] >= 3]
df = df[df["KT3"] >= 2]
df = df[df["KT1"] >= 3]
df = df[df["BC"] >= 4]

df.select_dtypes(include=["number"]).plot(
    kind="box",
    subplots=True,
    layout=(2, 5),
    figsize=(12, 8),
    sharex=False,
    sharey=False,
)

plt.suptitle("Boxplots of Each Numeric Column")
plt.show()

df.to_csv("processed_data.csv", index=False)  # Save processed data to CSV

# Conclusion
