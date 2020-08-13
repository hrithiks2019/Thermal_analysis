import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np


def get_object(filepathy):
    f = open('Support/labels.txt', 'r+')
    objects = [line[2:-1] for line in f.readlines()]
    model_path = 'Support/keras_model.h5'
    np.set_printoptions(suppress=True)
    model = tensorflow.keras.models.load_model(model_path)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(filepathy)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction_list = (list(model.predict(data)))[0]
    predicted_max = max((list(model.predict(data)))[0])
    try:
        idx = list(np.where(prediction_list == predicted_max))[0][0]
        predicted_object = objects[idx]
    except IndexError:
        predicted_object = 'Object Not Classified in the Trained Model'
    f.close()
    return predicted_object
