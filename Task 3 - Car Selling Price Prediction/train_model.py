import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("car data.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# Create Car Age exactly as in your notebook
df["Car_Age"] = 2018 - df["Year"]

# Features and target
X = df.drop(["Selling_Price", "Car_Name", "Year"], axis=1)
y = df["Selling_Price"]

# Categorical columns
categorical_cols = ["Fuel_Type", "Selling_type", "Transmission"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            categorical_cols
        )
    ],
    remainder="passthrough"
)

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train on complete dataset
model.fit(X, y)

# Save locally
joblib.dump(model, "car_price_model.pkl")

print("Model trained and saved successfully!")
print("Rows used:", len(X))