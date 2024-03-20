from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import csv
import matplotlib.pyplot as plt

model = load_model('modelo.keras')
scaler = MinMaxScaler()

def load_images_from_folder(image_folder):
    images = []
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.0 
        images.append(image)
    return images

def load_labels(path):
    with open(path, encoding='utf-8') as csv_file:
        niveles_colante = []
        reader = csv.DictReader(csv_file)
        for row in reader:
            niveles_colante.append(float(row['COLANTE']))
        return niveles_colante

image_folder = "./ImagenesTest"

test_images = load_images_from_folder(image_folder)

test_niveles_colante = load_labels("./Datos Test.csv")

X_test = np.array(test_images)

y_test = np.array(test_niveles_colante)
y_test = scaler.fit_transform(y_test.reshape(-1, 1))

# Predict the amount of adhesive for the new images
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test_transformed = scaler.inverse_transform(y_test)
plt.scatter(y_test_transformed, predictions)
plt.plot(np.linspace(min(y_test_transformed), max(y_test_transformed)), 
         np.linspace(min(y_test_transformed), max(y_test_transformed)), color='red')  # x=y line
plt.show()