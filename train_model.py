import os
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------
# 1. Load and prepare the dataset
# ---------------------------
titanic = sns.load_dataset("titanic")

# Select useful features
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = titanic[features]
y = titanic["survived"]

# Handle missing values (drop rows with missing data)
X = X.dropna()
y = y.loc[X.index]

# Encode categorical variables
X["sex"] = LabelEncoder().fit_transform(X["sex"])          # male=1, female=0
X["embarked"] = LabelEncoder().fit_transform(X["embarked"]) # C=0, Q=1, S=2

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------
# 2. Build a simple neural network
# ---------------------------
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# ---------------------------
# 3. Compile the model
# ---------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# 4. Train the model
# ---------------------------
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# ---------------------------
# 5. Save the model + scaler
# ---------------------------
os.makedirs("models", exist_ok=True)
model.save("models/titanic_nn_model.h5")

import joblib
joblib.dump(scaler, "models/titanic_scaler.pkl")

print("✅ Neural network trained and saved to models/titanic_nn_model.h5")
print("✅ Scaler saved to models/titanic_scaler.pkl")
