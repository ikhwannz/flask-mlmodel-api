from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model1 = tf.keras.models.load_model('subscribe_murnicampuran.hdf5') #murni dan tidak murni
model2 = tf.keras.models.load_model('subscribe3.hdf5') #fresh dan tidak fresh

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file)

    image1 = preprocess_image(image, (150, 150))
    image2 = preprocess_image(image, (150, 150))

    prediction1 = model1.predict(np.expand_dims(image1, axis=0))[0].tolist()
    prediction2 = model2.predict(np.expand_dims(image2, axis=0))[0].tolist()


    combined_prediction = {
        'prediction1': prediction1,
        'prediction2': prediction2,
    }

    hasil1 = ""
    hasil2 = ""

    value1 = combined_prediction["prediction1"][0]
    value2 = combined_prediction["prediction1"][1]
    value3 = combined_prediction["prediction2"][0]
    value4 = combined_prediction["prediction2"][1]

    if value1 > value2:
        hasil1 = "murni"
    else:
        hasil1 = "campuran"

    if value3 > value4:
        hasil2 = "fresh"
    else:
        hasil2 = "spoiled"

    hasil_akhir = "Daging ini merupakan " +"daging "+hasil1 +" dan daging "+ hasil2

    return jsonify({'predict': hasil_akhir})

def preprocess_image(image, size):
    image = image.resize(size)
    image = np.array(image) / 255.0
    image = (image - 0.5) / 0.5
    return image

if __name__ == '_main_':
    app.run()