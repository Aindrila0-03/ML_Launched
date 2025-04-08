from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available, using CPU.")

# Load model once globally
try:
    model = tf.keras.models.load_model('cifar10_model.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    logging.error(f"Error loading model: {e}")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Flask setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB upload limit

# Use an absolute path for the upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    is_correct = None
    img_path = ""

    if request.method == 'POST':
        img_file = request.files.get('image')
        if img_file:
            try:
                # Validate file type
                if not img_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return "Unsupported file type. Please upload a PNG or JPG image.", 400

                # Save the uploaded image
                path = os.path.join(UPLOAD_FOLDER, img_file.filename)
                img_file.save(path)

                # Load and preprocess the image
                img = image.load_img(path, target_size=(32, 32))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Make prediction
                pred = model.predict(img_array)
                predicted_class = class_names[np.argmax(pred)]
                prediction = f"Prediction: {predicted_class}"

                # Optional label check from filename (e.g., cat.jpg → cat)
                actual_label = img_file.filename.split('.')[0].lower()
                is_correct = (predicted_class.lower() == actual_label)

                img_path = os.path.join('static/uploads', img_file.filename)

            except Exception as e:
                logging.error(f"Error processing image: {e}")
                return "An error occurred while processing the image.", 500

    return render_template('index.html',
                           prediction=prediction,
                           img_path=img_path,
                           is_correct=is_correct)

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled Exception: {e}")
    return "An internal error occurred. Please try again later.", 500

if __name__ == '__main__':
    # Uncomment the following line to force CPU usage
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    app.run(debug=True)
