# Import required libraries
from flask import Flask, request, jsonify, render_template  # Flask for web app and API
import numpy as np  # For numerical operations and creating arrays
import tensorflow as tf  # TensorFlow for loading and using your neural network model

# ----------------------------
# Load the Keras neural network model
# ----------------------------
# This loads the model we previously trained and saved in HDF5 (.h5) format.
# Make sure the path matches where your model file is located.
model = tf.keras.models.load_model("models/titanic_nn_model.h5")

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Define route for homepage
# ----------------------------
@app.route("/")
def home():
    # Renders the HTML page "index.html" when user visits the root URL
    return render_template("index.html")

# ----------------------------
# Define route for predictions
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Sex is already numeric (0 or 1) from frontend
        sex = int(data["Sex"])
        
        features = np.array([[data["Pclass"], data["Age"], sex]], dtype=float)

        prob = model.predict(features)[0][0]

        if prob >= 0.5:
            output = f"✅ Survived ({prob*100:.1f}%)"
        else:
            output = f"❌ Did Not Survive ({(1-prob)*100:.1f}%)"

        return jsonify({"prediction": output, "probability": float(prob)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run the Flask app
# ----------------------------
# The app will start a local server at http://127.0.0.1:5000/ (default)
if __name__ == "__main__":
    app.run(debug=True)  # debug=True reloads automatically on code changes
