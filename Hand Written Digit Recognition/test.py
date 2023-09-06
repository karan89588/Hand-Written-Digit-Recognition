import cv2
import tensorflow as tf
import numpy as np
import pickle

modelP = pickle.load(open("trained_model.pickle", "rb"))
model = modelP["model"]
img = cv2.imread("8.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resizeImg = cv2.resize(grayImg, (28, 28), interpolation=cv2.INTER_AREA)
normalizeImg = tf.keras.utils.normalize(resizeImg, axis=1)
resizedNormalizedImg = np.array(normalizeImg).reshape(-1, 28, 28, 1)
prediction = model.predict(resizedNormalizedImg)
print(np.argmax(prediction))
