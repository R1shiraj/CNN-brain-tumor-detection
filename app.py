from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image




# Define a flask app
app = Flask(__name__)




# Define constants
IMAGE_SIZE = 128
NUM_CLASSES = 4  # Assuming two classes: Tumor and NoTumor

# Load the saved weights
model = load_model('C:/Users/Rishiraj/OneDrive/Desktop/my_model.h5')

# Function to predict on a single image
def predict_image(image_path, model):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB') #Convert to RGB to ensure 3 channels
    x = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))

    # Ensure the image has the correct shape and channels
    if len(x.shape) == 2:  # If the image is grayscale, convert it to RGB format
        x = np.stack((x,) * 3, axis=-1)
    
    x = x.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    x = x / 255.0  # Normalizing the image
    # Perform the prediction
    res = model.predict_on_batch(x)
    return res
    # classification = np.argmax(res, axis=-1)[0]

    # # Output the prediction
    # labels = ["Glioma", "Meningioma", "NO Tumor", "Pituitary"]
    # print(f"{res[0][classification] * 100:.2f}% Conclusion: {labels[classification]}")


# Example usage
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\meningioma\Te-me_0097.jpg")
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\pituitary\Te-pi_0013.jpg")
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\notumor\Te-no_0013.jpg")
# predict_image(r"C:\Users\Rishiraj\OneDrive\Desktop\DatasetBrainMRI\Testing\glioma\Te-gl_0013.jpg")

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_image(file_path, model)
        classification = np.argmax(preds, axis=-1)[0]

        labels = ["Glioma", "Meningioma", "NO Tumor", "Pituitary"]
        return (f"{preds[0][classification] * 100:.2f}% Conclusion: {labels[classification]}")
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # #**************************************************************************************************
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result
        #*************************************************************************************************

    return None


if __name__ == '__main__':
    app.run(debug=True)
