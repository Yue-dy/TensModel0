import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value


# def prepare(filepath):
#     IMG_SIZE = 100  # 50 in txt-based
#     img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
#     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath)
    (b,g,r) = cv2.split(img_array)
    img_array = cv2.merge([r,g,b])
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("64x3-CNN.model")
prediction = []
for i in range(10):
    path = os.path.join('test/',str(i) +'.jpg')
    pred = CATEGORIES[int(round(model.predict(prepare(path))[0][0]))]
    prediction.append(pred)
    plt.title(pred)
    a = prepare(path).reshape(100,100,3)
    plt.imshow(a,cmap=plt.cm.binary)
    plt.show()
