import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Model path (adjust based on your actual location)
model_path = r"E:\Git Uploads\Deep-Learning\Fashion MNIST\app\trained_model\trained_model.keras"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded images
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))  # Resize to match model input
    img = img.convert('L')     # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return img_array

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an Image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # Display the uploaded image (optional)
    st.image(image, caption='Uploaded Image', use_column_width=True) 
    
    if st.button('Classify'):
        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_image)

        # Make a prediction
        result = model.predict(img_array)
        predicted_class = np.argmax(result)
        prediction = class_names[predicted_class]

        st.success(f'Prediction: {prediction}') 
