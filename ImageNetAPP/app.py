from inference import get_prediction, transform_image
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(img_bytes)
        print(class_id, class_name)
        return jsonify({ 'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()