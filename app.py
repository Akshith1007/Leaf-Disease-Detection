from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model(r"C:\Users\akshi\Downloads\Projects\plant_disease\leaf_disease_v2.h5")

# Set folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels for prediction
class_labels = ['Apple - Apple scab', 'Apple - Black rot', 'Apple - Cedar apple rust', 'Apple - Healthy',
                'Blueberry - Healthy', 'Cherry - Powdery mildew', 'Cherry - Healthy',
                'Corn - Cercospora leaf spot (Gray leaf spot)', 'Corn - Common rust', 'Corn - Northern Leaf Blight',
                'Corn - Healthy', 'Grape - Black rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf blight',
                'Grape - Healthy', 'Orange - Citrus greening (Haunglongbing)', 'Peach - Bacterial spot', 
                'Peach - Healthy', 'Pepper - Bacterial spot', 'Pepper - Healthy', 
                'Potato - Early blight', 'Potato - Late blight', 'Potato - Healthy', 
                'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery mildew', 
                'Strawberry - Leaf scorch', 'Strawberry - Healthy', 
                'Tomato - Bacterial spot', 'Tomato - Early blight', 'Tomato - Late blight',
                'Tomato - Leaf Mold', 'Tomato - Septoria leaf spot', 
                'Tomato - Spider mites (Two-spotted spider mite)', 'Tomato - Target Spot',
                'Tomato - Tomato Yellow Leaf Curl Virus', 'Tomato - Tomato mosaic virus', 'Tomato - Healthy']

# Allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If user does not select a file, the browser submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the uploaded image and prepare it for prediction
            img = Image.open(filepath)
            img = img.resize((224, 224))  # Resize image for your model input size
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = img / 255.0  # Normalize the image

            # Make prediction using the loaded model
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            result = class_labels[predicted_class]

            # Pass the uploaded file path and prediction result to the template
            return render_template('index.html', uploaded_image=filename, result=result)

    return render_template('index.html', uploaded_image=None, result=None)

if __name__ == "__main__":
    app.run(debug=True)
