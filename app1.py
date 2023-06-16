import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image 

st.title("Dog Cat Classifier")

# Step1: Load the model
model=load_model("cats_dogs_small_2.h5")

# Step2: Load the image 
uploaded_file=st.file_uploader('Choose the database', accept_multiple_files=False)
if uploaded_file is not None:
    file=uploaded_file
else:
    file='image.jpg'

image=Image.open(file)
st.image(image)

# Step3: Preprocess the loaded image
img=load_img(file, target_size=(150, 150))
img_arr=img_to_array(img)
img_arr_ready=np.expand_dims(img_arr, axis=0)

# Step4 : Get the prediction
prediction=int(model.predict(img_arr_ready)[0][0])
if prediction==1:
    st.subheader('Its a dog')
else:
    st.subheader("Its a cat")
