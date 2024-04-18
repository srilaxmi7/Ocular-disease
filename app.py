import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model/model2.keras')  # Load your model file
    return model

model = load_model()

# Define function to preprocess image
def preprocess_image(image):
    # Preprocess your image here (resize, normalize, etc.)
    return image

# Define function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Define your Streamlit app
def main():
    st.title('Eye Disease Prediction')

    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        prediction = predict(image)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
