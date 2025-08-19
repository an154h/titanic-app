import numpy as np
import tensorflow as tf

# ----------------------------
# Create a tiny dataset
# ----------------------------
# X contains features for each passenger: [Pclass, Age, Sex]
# Pclass: 1 = 1st class, 2 = 2nd class, 3 = 3rd class
# Age: passenger's age
# Sex: 1 = male, 0 = female
X = np.array([
    [1, 22, 1],
    [3, 38, 0],
    [2, 26, 1],
    [3, 35, 0],
    [1, 54, 0],
    [2, 2, 1],
    [3, 27, 0],
    [1, 40, 1]
], dtype=float)

# y contains the target variable: Survived (1) or Did Not Survive (0)
y = np.array([0, 1, 1, 1, 0, 1, 0, 1], dtype=float)

# ----------------------------
# Build a tiny neural network
# ----------------------------
model = tf.keras.Sequential([
    # First layer: Dense layer with 8 neurons, ReLU activation
    # input_shape=(3,) because each input has 3 features
    tf.keras.layers.Dense(8, input_shape=(3,), activation='relu'),
    
    # Second layer: Dense layer with 4 neurons, ReLU activation
    tf.keras.layers.Dense(4, activation='relu'),
    
    # Output layer: 1 neuron with sigmoid activation
    # Sigmoid outputs a probability between 0 and 1
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ----------------------------
# Compile the model
# ----------------------------
# optimizer='adam' → a popular optimization algorithm for training neural networks
# loss='binary_crossentropy' → suitable for binary classification (survived or not)
# metrics=['accuracy'] → track accuracy during training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# Train the model
# ----------------------------
# model.fit trains the model on X and y
# epochs=100 → train for 100 iterations over the entire dataset
# verbose=0 → suppress training output (quiet mode)
model.fit(X, y, epochs=100, verbose=0)

# ----------------------------
# Save the trained model
# ----------------------------
# Saves the neural network in HDF5 format (.h5) so it can be loaded later in Flask or elsewhere
model.save("models/titanic_nn_model.h5")

print("✅ Neural network model saved as models/titanic_nn_model.h5")
