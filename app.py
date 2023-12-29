# app.py
from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import joblib


app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("my_finetuned_model.h5")

# Load the saved LabelEncoder
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = np.fromstring(file.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            return render_template("index.html", prediction=predicted_label)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
