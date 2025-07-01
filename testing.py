import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

def load_label_dict(pkl_path):
    with open(pkl_path, 'rb') as f:
        label_dict = pickle.load(f)
    return label_dict

def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_traffic_sign(model_path, label_dict_path, test_image_path, temperature=1.0):
    # Load model
    model = load_model(model_path)

    # Load label dictionary
    label_dict = load_label_dict(label_dict_path)

    # Preprocess the test image
    img = preprocess_image(test_image_path)

    # Predict probabilities
    prediction = model.predict(img)

    # Apply temperature scaling
    prediction = np.exp(np.log(prediction) / temperature)
    prediction /= np.sum(prediction)

    # Get predicted class index
    predicted_class = np.argmax(prediction)

    # Reverse label_dict to get label from index
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    predicted_label = reverse_label_dict[predicted_class]

    confidence = np.max(prediction) * 100

    print(f"Predicted Traffic Sign: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    # Paths to your model, label dict, and test image on your PC
    model_path = r"traffic_sign_cnn.keras"           # Update this path
    label_dict_path = r"label_dict.pkl"              # Update this path
    test_image_path = r"C:\Users\aditya\Downloads\y_intersection-trafficsign.jpg" # Update this path

    predict_traffic_sign(model_path, label_dict_path, test_image_path)
