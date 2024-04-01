import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import csv
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from joblib import dump



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

train_image_folder = "./Imagenes"
test_image_folder = "./ImagenesTest"

train_images = load_images_from_folder(train_image_folder)
test_images = load_images_from_folder(test_image_folder)

train_niveles_colante = load_labels("./Datos Aprendizaje.csv")
test_niveles_colante = load_labels("./Datos Test.csv")

X_train = np.array(train_images)
X_test = np.array(test_images)

y_train = np.array(train_niveles_colante)
y_test = np.array(test_niveles_colante)



# Create a scaler object
scaler = MinMaxScaler()

# Fit the scaler to the training labels and transform
y_train = scaler.fit_transform(y_train.reshape(-1, 1))

# Save the fitted scaler for later use
dump(scaler, 'scaler.joblib') 

# Transform the test labels
y_test = scaler.transform(y_test.reshape(-1, 1))
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=Adam(learning_rate=0.00005), loss='mean_squared_error')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stop])
model.save("modelo.keras")

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test_transformed = scaler.inverse_transform(y_test)
plt.scatter(y_test_transformed, predictions)
plt.plot(np.linspace(min(y_test_transformed), max(y_test_transformed)), 
         np.linspace(min(y_test_transformed), max(y_test_transformed)), color='red')  # x=y line
plt.show()