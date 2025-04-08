from PIL import Image
from flask import Flask, render_template, request

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Tells TensorFlow to use CPU only


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np



# Flask setup
app = Flask(__name__)

# Load model
model = load_model('cifar10_model.keras')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    is_correct = None
    img_path = ""

    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            path = os.path.join(UPLOAD_FOLDER, img_file.filename)
            img_file.save(path)

            img = image.load_img(path, target_size=(32, 32))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            prediction = f"Prediction: {predicted_class}"

            # Extract actual label from filename (e.g., "dog.jpg" → "dog")
            actual_label = img_file.filename.split('.')[0].lower()
            is_correct = (predicted_class.lower() == actual_label)

            img_path = os.path.join('static', img_file.filename)
            return render_template('index.html', prediction=prediction, img_path=img_path, is_correct=is_correct)

    return render_template('index.html', prediction=prediction, img_path=img_path, is_correct=is_correct)

if __name__ == '__main__':
    app.run()
