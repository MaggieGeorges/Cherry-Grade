from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("best (1).pt")  # Load your trained model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            # Save uploaded image
            img_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
            img_file.save(img_path)

            # Run inference
            results = model(img_path)
            annotated = results[0].plot()

            # Save result
            result_path = os.path.join(RESULT_FOLDER, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(result_path, annotated)

            return render_template("index.html", result_image=result_path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)