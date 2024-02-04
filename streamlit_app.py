import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

@st.cache_resource
def load_ml_model():
    return load_model('digit_recogniser4.h5')


st.title("Hand written digit recognizer")
st.write("Write a digit in the canvas below. Convoltional Neural network(CNN) will detect the digit.")

drawing_mode =  "freedraw"
stroke_width = 20
stroke_color = "#ffffff"
bg_color = "#000000"

con1 = st.container(border=True)
col1,col2 = con1.columns((1,1))
with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=None,
        height=250,
        width=250,
        drawing_mode=drawing_mode,
        point_display_radius= 0,
        key="canvas",
    )

if canvas_result.image_data is not None:
    # st.image(canvas_result.image_data)
    # print(type(canvas_result.image_data))
    im = Image.fromarray(canvas_result.image_data)
    im= im.convert("L")
    im = im.resize((28,28))
    # st.image(im)
    model = load_ml_model()
    # print(np.array(im).shape)
    prediction = model.predict(np.array(im).reshape(1,28,28,1))
    col2.write("Predicted result:")
    col2.header(np.argmax(prediction))
    col2.write(prediction)
