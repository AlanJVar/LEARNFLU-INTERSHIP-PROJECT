import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("Customer Churn.csv")
print(df.head())
print(df.info())
print(df.describe())

# Missing values visualization
# This plot will give a screen with a singular color as there are no null values
# This was the conclusion I reached upon plotting this graph
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Values Heatmap")
plt.show()


# I have made this plot to show how last column i.e churn is spread/distributed over the entire data
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df, palette="coolwarm")
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Histogram
df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Pairplot
sns.pairplot(df, hue="Churn", diag_kind="kde")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Convert categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop(columns=["Churn"]))

# Final dataset
X = pd.DataFrame(scaled_data, columns=df.drop(columns=["Churn"]).columns)
y = df["Churn"]
