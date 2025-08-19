import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib  # We'll still save a scaler if needed

# ---------------------------
# 1. Prepare the dataset
# ---------------------------
data = {
    "Pclass": [1, 3, 2, 3, 1, 2, 3, 1],
    "Age":    [22, 38, 26, 35, 54, 2, 27, 40],
    "Sex":    [1, 0, 1, 0, 0, 1, 0, 1],   # 1 = male, 0 = female
    "Survived": [0, 1, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[["Pclass", "Age", "Sex"]].values
y = df["Survived"].values

# ---------------------------
# 2. Build a simple neural network
# ---------------------------
model = Sequential([
    Dense(8, input_dim=3, activation='relu'),  # hidden layer with 8 neurons
    Dense(4, activation='relu'),               # hidden layer with 4 neurons
    Dense(1, activation='sigmoid')             # output layer (binary classification)
])

# ---------------------------
# 3. Compile the model
# ---------------------------
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# 4. Train the model
# ---------------------------
model.fit(X, y, epochs=200, batch_size=1, verbose=1)

# ---------------------------
# 5. Save the model
# ---------------------------
# Save the neural network in HDF5 format
model.save("models/titanic_nn_model.h5")

print("✅ Neural network trained and saved to models/titanic_nn_model.h5")
