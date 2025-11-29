#FRUIT AND VEG MODEL 

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


# Loading the model
model = load_model('fruit_model.h5')

def process_image(img): 
    img = img.resize((32, 32))
    img = np.array(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    return img 

# Title and description
st.title(":kiwifruit: Fruit and Veg Detection  :kiwifruit: ")
st.write("Upload a Picture and The Model Will Guess What Type It Is")

# Image upload area
file = st.file_uploader('Select a Picture', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Image')
    
    # Process the image
    image = process_image(img)
    
    # Predict
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Class names
    class_names = ['corn', 'cabbage', 'cauliflower', 'capsicum', 'cucumber', 'carrot', 'banana',
                    'turnip', 'kiwi', 'beetroot', 'raddish', 'grapes', 'lemon', 'potato', 'spinach', 
                    'pear', 'sweetpotato', 'orange', 'eggplant', 'jalepeno', 'paprika', 'watermelon',
                      'chilli pepper', 'onion', 'tomato', 'peas', 'bell pepper', 'soy beans', 'garlic', 
                      'lettuce', 'apple', 'mango', 'pomegranate', 'sweetcorn', 'ginger', 'pineapple']
    
    # Show prediction result
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")