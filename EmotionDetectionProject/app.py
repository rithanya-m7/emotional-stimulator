from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load trained model
model = load_model("emotion_detection_model.h5")

# Emotion labels
emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


@app.route('/', methods=['GET', 'POST'])
def index():

    emotion = ""

    if request.method == 'POST':

        file = request.files['image']

        if file:

            # Convert uploaded file into OpenCV format
            file_bytes = np.frombuffer(file.read(), np.uint8)

            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            # Load face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Detect faces
            faces = face_cascade.detectMultiScale(img, 1.3, 5)

            if len(faces) > 0:

                x, y, w, h = faces[0]

                # Crop face
                face = img[y:y+h, x:x+w]

                # Resize image
                face = cv2.resize(face, (48, 48))

                # Normalize image
                image_array = face / 255.0

                # Reshape for model
                image_array = image_array.reshape(1, 48, 48, 1)

                # Predict emotion
                prediction = model.predict(image_array)

                emotion = emotions[np.argmax(prediction)]

            else:
                emotion = "No Face Detected"

    return render_template('index.html', emotion=emotion)


if __name__ == '__main__':
    app.run(debug=True)