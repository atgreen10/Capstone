import streamlit as st
import tensorflow as tf
import numpy as np


# tensorflow model prediction

def prediction(test_image):
    model = tf.keras.models.load_model('trained_model/trained_disease_detection.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256, 3))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # converts single image to a batch of images.
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result = np.argmax(prediction)
    return result


# sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Find Your Way:", ['Home', 'About', 'Detect'])

if app_mode == 'Home':
    st.header("Plant Disease Detective")
    image_path = "plantdetectives.jpg"
    st.image(image_path, use_column_width=True)

if app_mode == 'About':
    st.header("About the App")
    st.markdown("""
        This application is more about the end than it is the beginning. This is the ***Plant Disease Detective*** AI 
    application, developed by Austin Green. This program is the culmination of all the learning I did at WGU and I am 
    presenting this AI program as my Capstone Final Project. I'll be honest, when I wrote about this program in C951,
    I never thought I would actually be writing the code to make this an actual, working program, and yet here I am, or better yet,
    here the program is!    
    """)

if app_mode == 'Detect':
    st.header("Let the Detective Go To Work")
    uploaded_image = st.file_uploader("Upload your image here", type='jpg')
    if (st.button("Start Detective work")):
        st.image(uploaded_image, use_column_width=True)
        results = prediction(uploaded_image)
        class_name = ['Apple___Apple_scab',
                      'Apple___Black_rot',
                      'Apple___Cedar_apple_rust',
                      'Apple___healthy',
                      'Blueberry___healthy',
                      'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy',
                      'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_',
                      'Corn_(maize)___Northern_Leaf_Blight',
                      'Corn_(maize)___healthy',
                      'Grape___Black_rot',
                      'Grape___Esca_(Black_Measles)',
                      'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy',
                      'Orange___Haunglongbing_(Citrus_greening)',
                      'Peach___Bacterial_spot',
                      'Peach___healthy',
                      'Pepper,_bell___Bacterial_spot',
                      'Pepper,_bell___healthy',
                      'Potato___Early_blight',
                      'Potato___Late_blight',
                      'Potato___healthy',
                      'Raspberry___healthy',
                      'Soybean___healthy',
                      'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch',
                      'Strawberry___healthy',
                      'Tomato___Bacterial_spot',
                      'Tomato___Early_blight',
                      'Tomato___Late_blight',
                      'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot',
                      'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot',
                      'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                      'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("We have detected that this plant has signs of {}".format(class_name[results]))
