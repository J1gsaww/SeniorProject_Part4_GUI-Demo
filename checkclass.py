from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
import numpy as np

model_path = 'trained_model.h5'
model = load_model(model_path)

target_img_shape = (100, 100)

def preprocess_image(img_path):
    img = load_img(img_path, target_size=target_img_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_class(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

img_path = 'data\\data\\test\\ญ\\ญ_48.png'
predicted_class = predict_class(img_path)
print(f'Predicted Class: {predicted_class}')
