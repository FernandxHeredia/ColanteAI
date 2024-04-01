import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load

def load_images_from_folder(image_folder):
    images = []
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.0 
        images.append((image, image_file))  # Return a tuple of image and filename
    return images

def predict_and_plot(model_path, image_folder, scaler_path):
    # Load the model
    model = load_model(model_path)

    # Load the images and filenames
    images_and_filenames = load_images_from_folder(image_folder)

    # Separate the images and filenames into two lists
    images, filenames = zip(*images_and_filenames)

    # Convert the list of images to a numpy array
    X = np.array(images)

    # Load the scaler
    scaler = load(scaler_path)

    # Make predictions
    predictions = model.predict(X)

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(predictions)

    # Plot the predictions
    plt.figure(figsize=(10, 5))
    for i, prediction in enumerate(predictions):
        plt.plot(i, prediction, 'o')
        plt.text(i, prediction, filenames[i])
    plt.title('Predictions')
    plt.xlabel('Image Index')
    plt.ylabel('Prediction')
    plt.grid(True)
    plt.show()

# Call the function with the path to the model, the image folder, and the scaler
predict_and_plot("modelo.keras", "./ImagenesTest", "scaler.joblib")
