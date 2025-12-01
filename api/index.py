from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import pandas as pd # Example for data handling

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "pickles", "iris_log_reg.pkl")
IRIS_JSON_PATH = os.path.join(BASE_DIR, "datasets", "iris.json")

with open(MODEL_PATH, "rb") as f:
    iris_model = pickle.load(f)

iris_species = ["Setosa", "Versicolor", "Virginica"]

@app.route("/", methods=["GET"])
def hello_world():
    return "<p>Hello, World!</p>"

# API call to handle Iris model requests
@app.route('/predict-iris', methods=['POST'])
def predict_iris():
    data = request.get_json(force=True)
    # Assuming input data is a dictionary matching model's features
    df = pd.DataFrame([data])
    prediction = iris_model.predict(df)  # returns an index for the iris_species list
    index = int(prediction[0])
    return jsonify({'prediction': iris_species[index]})

    # return jsonify({'prediction': prediction.tolist()})

# API call to fetch Iris dataset in JSON format
@app.route('/iris-data', methods=['GET'])
def get_iris_data():
    with open('datasets/iris.json') as iris_data:
        iris_json = json.load(iris_data)
        return iris_json

# House Price Section
# API call to handle house price prediction requests
@app.route('/predict-house', methods=['POST'])
def predict_house():
    return jsonify({'prediction': 'TEST!'})