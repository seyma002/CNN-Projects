#INDIAN BIRD MODEL


import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('bird_model.h5')

def process_image(img): 
    img = img.resize((32, 32))
    img = np.array(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    return img 


st.title(" :star2: :bird: Indian Bird :bird: :star2: ")
st.write("Upload a Picture and The Model Will Guess What Type It Is")


file = st.file_uploader('Select a Picture', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Image')
    
    
    image = process_image(img)
    
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    
    class_names = [
        'Hoopoe', 'Cattle Egret', 'Gray Wagtail', 'House Crow', 'White-Breasted Kingfisher', 
        'Common Rosefinch', 'Sarus Crane', 'Northern Lapwing', 'Coppersmith Barbet', 
        'Forest Wagtail', 'White Wagtail', 'Brown-Headed Barbet', 'Ruddy Shelduck', 
        'Asian Green Bee-Eater', 'White-Breasted Waterhen', 'Common Tailorbird', 
        'Indian Grey Hornbill', 'Indian Roller', 'Rufous Treepie', 'Common Myna', 
        'Jungle Babbler', 'Indian Pitta', 'Red-Wattled Lapwing', 'Indian Peacock', 
        'Common Kingfisher'
    ]
    
    
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")