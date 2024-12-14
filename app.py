import os
import librosa
import numpy as np
from flask import Flask, request, render_template
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# Define the path to save the trained model
MODEL_PATH = 'deepfake_audio_model.pkl'

# Define the base path for the uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path where the audio data resides (KAGGLE/AUDIO folder)
AUDIO_DATA_PATH = r"C:\Users\91770\.cache\kagglehub\datasets\birdy654\deep-voice-deepfake-voice-recognition\versions\2\KAGGLE\AUDIO"
fake_folder = os.path.join(AUDIO_DATA_PATH, 'FAKE')
real_folder = os.path.join(AUDIO_DATA_PATH, 'REAL')

# Function to process the audio file and extract features (e.g., MFCC)
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    mfcc = np.mean(mfcc, axis=1)  # Take the mean of the MFCCs for a fixed-length feature vector
    return mfcc

# Function to train and save the model
def train_and_save_model():
    X = []  # Features
    y = []  # Labels (1 for real, 0 for fake)

    # Ensure the FAKE and REAL folders exist
    if not os.path.exists(fake_folder) or not os.path.exists(real_folder):
        print("Error: FAKE or REAL folder does not exist.")
        return

    # Check and list files in 'FAKE' folder
    fake_files = [f for f in os.listdir(fake_folder) if f.endswith('.wav')]
    if not fake_files:
        print(f"No '.wav' files found in {fake_folder}")
        return

    # Check and list files in 'REAL' folder
    real_files = [f for f in os.listdir(real_folder) if f.endswith('.wav')]
    if not real_files:
        print(f"No '.wav' files found in {real_folder}")
        return

    print(f"Found {len(fake_files)} fake files and {len(real_files)} real files.")

    # Load audio files from the FAKE and REAL folders and extract features
    for label, folder in enumerate([real_folder, fake_folder]):
        for file_name in os.listdir(folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder, file_name)
                features = process_audio(file_path)
                X.append(features)
                y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("Error: No audio features extracted.")
        return

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a classifier (e.g., RandomForest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model to a file
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

# Check if a model already exists, otherwise train a new one
if not os.path.exists(MODEL_PATH):
    print("No pre-trained model found. Training a new model...")
    train_and_save_model()

# Load the trained model
model = joblib.load(MODEL_PATH)

# Route for the main page
@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return 'No file part', 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and file.filename.endswith('.wav'):  # Update this check to '.wav'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process the uploaded audio file
        features = process_audio(file_path)
        
        # Make prediction using the trained model
        prediction = model.predict([features])
        
        # Map the model prediction to a label
        result = "REAL" if prediction == 1 else "FAKE"
        
        # Display the result on the page
        return render_template('upload.html', prediction=result)
    
    return 'Invalid file format. Only WAV files are allowed.', 400

# Route to predict using audio files in the AUDIO_DATA_PATH
@app.route('/predict_from_data')
def predict_from_data():
    fake_folder = os.path.join(AUDIO_DATA_PATH, 'FAKE')
    real_folder = os.path.join(AUDIO_DATA_PATH, 'REAL')

    # Check and list files in 'FAKE' folder
    if os.path.exists(fake_folder):
        fake_files = os.listdir(fake_folder)
    else:
        fake_files = []

    # Check and list files in 'REAL' folder
    if os.path.exists(real_folder):
        real_files = os.listdir(real_folder)
    else:
        real_files = []

    # For demonstration, process the first fake and real file for prediction
    predictions = {}

    if fake_files:
        fake_file_path = os.path.join(fake_folder, fake_files[0])
        fake_features = process_audio(fake_file_path)
        fake_prediction = model.predict([fake_features])
        predictions['FAKE'] = "REAL" if fake_prediction == 1 else "FAKE"

    if real_files:
        real_file_path = os.path.join(real_folder, real_files[0])
        real_features = process_audio(real_file_path)
        real_prediction = model.predict([real_features])
        predictions['REAL'] = "REAL" if real_prediction == 1 else "FAKE"

    return render_template('upload.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
