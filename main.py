from flask import Flask, render_template, request
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import os
import shutil, shap
import pickle
import numba



app = Flask(__name__)
model = pickle.load(open("model_xgboost_31.pkl", "rb"))
# Inisialisasi objek explainer SHAP dengan model yang sudah dilatih
explainer = shap.Explainer(model)

def extract_features(images):
    features_list = []
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
        features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features_list.append(features)
    return np.array(features_list)

@app.route('/')
def home():
    return render_template('index_coba.html')

    # Get the uploaded images from the request
    images = request.files.getlist('images')

    # Save the uploaded images to a temporary folder
    temp_folder = 'temp_images'
    os.makedirs(temp_folder, exist_ok=True)
    image_paths = []
    for img in images:
        img_path = os.path.join(temp_folder, img.filename)
        img.save(img_path)
        image_paths.append(img_path)

    # Extract features from the uploaded images
    features = extract_features(image_paths)

    # Perform further processing with the extracted features
    # ...

    # Remove the temporary folder and images
    shutil.rmtree(temp_folder)

    return 'Image processing completed'

@app.route("/predict", methods=["POST"])
def predict():
    images = request.files.getlist('images')

    # Save the uploaded images to a temporary folder
    temp_folder = 'temp_images'
    os.makedirs(temp_folder, exist_ok=True)
    image_paths = []
    additional_features_list = []
    for img in images:
        img_path = os.path.join(temp_folder, img.filename)
        img.save(img_path)
        image_paths.append(img_path)

   # Get the additional form values for the current image
        harga = float(request.form.get('harga'))
        susunan = float(request.form.get('susunan'))
        bahan = float(request.form.get('bahan'))

        # Create the additional features array for the current image
        additional_features = np.array([[harga, susunan, bahan]], dtype=np.float32)
        additional_features_list.append(additional_features)

    # Extract features from the uploaded images
    features = extract_features(image_paths)

    # Convert the additional features list to a numpy array
    additional_features = np.concatenate(additional_features_list, axis=0)

    # Check the shape of the features array
    if features.shape[0] != additional_features.shape[0]:
        return 'Feature shape mismatch. Please make sure the number of features extracted from the images matches the expected shape.'

    # Combine the features and additional features arrays
    combined_features = np.concatenate((features, additional_features), axis=1)
    # Apply StandardScaler to scale all input features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    feature_names = scaler.get_feature_names_out()   

    # Calculate SHAP values for the scaled features
    shap_values = explainer(scaled_features)

    # Get the feature importance values for each image
    feature_importance = np.abs(shap_values.values).mean(axis=0)

    # Sort the feature importance values in descending order
    sorted_indices = np.argsort(feature_importance)[::-1]

    # Get the top N features and their names
    top_features_indices = sorted_indices[:8104] 
    # Get the top features and their names
    top_features_names = [feature_names[i] for i in top_features_indices if feature_names[i] in ['x8101', 'x8102', 'x8103']]
    
    prediction = model.predict(scaled_features)
    return render_template("index_coba.html", prediction_text="{}".format(prediction), top_features = top_features_names)

if __name__ == '__main__':
    app.run(debug=True)
