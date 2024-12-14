from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np
import os
import warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')

# Initialize Flask app
app = Flask(__name__)

# Model training code remains the same
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)
    return images, labels

# Load dataset
real_images, real_labels = load_images_from_folder(r"C:\Users\91770\Downloads\archive\real_and_fake_face\training_real", label=0)
fake_images, fake_labels = load_images_from_folder(r"C:\Users\91770\Downloads\archive\real_and_fake_face\training_fake", label=1)

# Combine and preprocess
images = np.array(real_images + fake_images)
labels = np.array(real_labels + fake_labels)
images = images / 255.0

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save('deepfake_model.h5')
print("Model training complete and saved as 'deepfake_model.h5'")

# Load the model for prediction
model = tf.keras.models.load_model('deepfake_model.h5')

# Define a function to test an uploaded image
def test_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    return "Fake" if prediction[0][0] > 0.5 else "Real"

# Flask routes for the web interface
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No MEDIA uploaded", 400
    file = request.files['image']
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)
    
    # Load and process the image for prediction
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image format", 400
    
    result = test_image(img)
    
    # Clean up by removing the uploaded image file
    os.remove(image_path)
    
    return render_template('upload.html', result=result)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)  # Ensure upload folder exists
    app.run(debug=True)
