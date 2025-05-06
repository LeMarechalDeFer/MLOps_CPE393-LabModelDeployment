from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    iris_model = pickle.load(f)
    
with open("housing_model.pkl", "rb") as f:
    housing_model = pickle.load(f)

HOUSING_COLUMNS = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "area_per_bed", "rooms_total",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]


@app.route("/")
def home():
    return "ML Model is Running: /predict-iris and /predict-housing available"


# def predict():
#     data = request.get_json()
#     input_features = np.array(data["features"]).reshape(1, -1)
#     prediction = model.predict(input_features)
#     return jsonify({"prediction": int(prediction[0])})


@app.route("/predict-iris", methods=["POST"])
def predict_iris():
    data = request.get_json()
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key. Expected format: {'features': [f1, f2, f3, f4]}"}), 400

    features = data["features"]
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Expected 4 numeric values for Iris features."}), 400

    input_array = np.array(features).reshape(1, -1)
    prediction = iris_model.predict(input_array)
    return jsonify({"prediction": int(prediction[0])})



@app.route("/predict-housing", methods=["POST"])
def predict_housing():
    data = request.get_json()

    if "features" not in data:
        return jsonify({
            "error": "Missing 'features' key.",
            "expected_format": {
                "features": [
                    "area", "bedrooms", "bathrooms", "stories", "parking",
                    "area_per_bed", "rooms_total",
                    "mainroad (0/1)", "guestroom (0/1)", "basement (0/1)",
                    "hotwaterheating (0/1)", "airconditioning (0/1)", "prefarea (0/1)",
                    "furnishingstatus ('unfurnished', 'semi-furnished', or 'furnished')"
                ]
            }
        }), 400

    features = data["features"]
    if not isinstance(features, list) or len(features) != len(HOUSING_COLUMNS):
        return jsonify({
            "error": f"Expected {len(HOUSING_COLUMNS)} features in this order.",
            "expected_format": HOUSING_COLUMNS
        }), 400

    try:
        input_df = pd.DataFrame([features], columns=HOUSING_COLUMNS)
        prediction = housing_model.predict(input_df)[0]
        return jsonify({"prediction": int(round(prediction))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/ask-housing", methods=["GET"])
def ask_housing():
    questions = [
        "What is the total area (in square feet)?",
        "How many bedrooms?",
        "How many bathrooms?",
        "How many stories (floors)?",
        "How many parking spots?",
        "What is the area per bedroom? (area / bedrooms)",
        "What is the total number of rooms? (bedrooms + bathrooms)",
        "Is there a main road access? (yes=1 / no=0)",
        "Is there a guest room? (yes=1 / no=0)",
        "Is there a basement? (yes=1 / no=0)",
        "Is hot water heating available? (yes=1 / no=0)",
        "Is air conditioning available? (yes=1 / no=0)",
        "Is it in a preferred area? (yes=1 / no=0)",
        "What is the furnishing status? (unfurnished / semi-furnished / furnished)"
    ]
    return jsonify({"questions": questions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)


