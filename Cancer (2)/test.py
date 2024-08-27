import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import math
st.set_option('deprecation.showfileUploaderEncoding', False)
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
st.title("Breast Cancer Detection using Convolutional Neural Network")

uploaded_file = st.file_uploader("Choose an image...", type="PNG")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    #image.show()
    st.image(image, caption='Uploaded Image.', width=300)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    data = np.rint(prediction)
    print(data)
    if(data[0][0]==1):
        st.write("Malignant")
    if(data[0][1]==1):
        st.write("Benign")

