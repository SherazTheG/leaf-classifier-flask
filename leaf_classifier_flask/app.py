from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model(os.path.join(BASE_DIR, 'leaf_model.h5'))
class_names = ['Bacteria', 'Fungi', 'Nematodes', 'Normal', 'Virus']
img_size = (224, 224)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess
            img = image.load_img(filepath, target_size=img_size)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions)
            pred_class = class_names[pred_index]
            is_diseased = pred_class != "Normal"

            return render_template('index.html',
                                   prediction=pred_class,
                                   is_diseased=is_diseased,
                                   filepath=f'static/uploads/{filename}')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
