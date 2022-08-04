import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the model
model = load_model('dog_breed.h5')

#Name of classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Afghan Hound', 'Entlebucher', 'Bernese Mountain Dog']

#Setting title of app
st.title("Predição de Raça de Doguinhos")
st.markdown("Envie uma foto do catioro")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button("Predict")

#On predict button click
if submit:

	if dog_image is not None:

		file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8) #Convert the file to an opencv image
		opencv_image = cv2.imdecode(file_bytes, 1)

		st.image(opencv_image, channels="BGR") #Displaying the image

		opencv_image = cv2.resize(opencv_image, (224,224)) #Resizing

		opencv_image.shape = (1,224,224,3) #Convert to onedimensional

		y_pred = model.predict(opencv_image) #Prediction

		st.title(str("A raça do doguinho é "+CLASS_NAMES[np.argmax(y_pred)]))