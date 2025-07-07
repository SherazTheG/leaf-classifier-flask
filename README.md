
# 🌿 Leaf Disease Classifier using Deep Learning and Flask

A lightweight web application that allows users—especially farmers—to upload images of plant leaves and identify if they are affected by common diseases. The classifier detects the presence of **Bacteria**, **Fungi**, **Nematodes**, **Viruses**, or confirms if the leaf is **Normal**.

---

## 📌 Features

- 🔍 Classifies leaf diseases into 5 categories.
- 🧠 Powered by a custom-built Convolutional Neural Network (CNN).
- 🧪 Trained with early stopping and dropout for better generalization.
- 💻 Flask-based web interface for simple and fast local use.

---

## 🧠 Model Architecture

A custom CNN built using Keras with the following layers:

```python
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])
```

### 📊 Performance (Training Snapshot)

- ✅ Peak Validation Accuracy: **~85%**
- 🔁 Early Stopping applied after patience of 3 epochs
- ⏱ Trained for up to 50 epochs, best weights restored

---

## 🗂 Dataset

The model was trained on a custom dataset containing images of leaves classified into:

1. Bacteria
2. Fungi
3. Nematodes
4. Virus
5. Normal

> *(Dataset not included in this repository — attach or upload separately if required)*

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Matplotlib
- HTML (Jinja templates)

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/leaf-classifier-flask.git
   cd leaf-classifier-flask
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the model file**
   Ensure `leaf_model.h5` and `class_names.pkl` are inside the `leaf_classifier_flask/` directory.

4. **Run the Flask app**
   ```bash
   cd leaf_classifier_flask
   python app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:5000` to use the app.

---

## 🖼️ Usage Instructions

- Upload a clear image of a leaf (`.jpg`, `.png`, etc.).
- The model will predict one of the 5 categories.
- The interface also shows whether the leaf is diseased or healthy.

---

## 📌 Project Structure

```
leaf-classifier-project/
├── dataset/                     # Dataset folder (optional)
├── modeltraining.ipynb          # Training notebook
├── requirements.txt             # Required libraries
├── README.md                    # You're reading this!
└── leaf_classifier_flask/
    ├── app.py                   # Flask backend
    ├── leaf_model.h5            # Trained Keras model
    ├── class_names.pkl          # Pickle file with class names
    ├── templates/
    │   └── index.html           # Frontend UI
    └── static/uploads/          # Uploaded leaf images
```

---

## 🚧 Future Scope

- Deploy on Render / Hugging Face Spaces for public access
- Integrate multilingual support for wider usability
- Add disease description and treatment suggestions

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🙌 Acknowledgments

Built by **Mohamed Sheraz M** 
Feel free to fork, improve, and contribute!
