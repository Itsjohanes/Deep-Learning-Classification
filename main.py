import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

# Load the saved model
model = load_model('model.h5')

# Load the image to be predicted
img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Predict the class of the image
pred = model.predict(img)
class_idx = np.argmax(pred)
if class_idx == 0:
    print("Cat")
elif class_idx == 1:
    print("Dog")
else:
    print("Other animal")
