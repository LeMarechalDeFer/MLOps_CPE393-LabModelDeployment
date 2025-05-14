



# MLOps - ML Model Deployment with Flask and Docker

This project demonstrates the deployment of machine learning models in a containerized Flask API using Docker.

## 📋 Project Structure

```plaintext
MLOps_CPE393-LabModelDeployment/
├── app/
│ ├── app.py
│ ├── housing_model.pkl
│ ├── iris_model.pkl
│ └── requirements.txt
├── Dockerfile
├── Housing.csv
├── train.py
├── README.md
└── homework/
```

## 🚀 Setup and Installation

### Local Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
3. Train models (if needed):
   ```bash
   python train.py
   ```
4. Launch the API:
   ```bash
   python app/app.py
   ```

### Using Docker (Recommended)

1. Build the image:
   ```bash
   docker build -t ml-model .
   ```
2. Run the container:
   ```bash
   docker run -p 9000:9000 ml-model
   ```

## 🔍 API Usage

### Health Check

- **Endpoint**: `/health`
- **Method**: GET
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### Iris Classification

- **Endpoint**: `/predict-iris`
- **Method**: POST
- **Payload** (example):
  ```json
  {
    "features": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 3.4, 5.4, 2.3]
    ]
  }
  ```
- **Response**:
  ```json
  {
    "confidences": [1.0, 0.99],
    "predictions": [0, 2]
  }
  ```

### Housing Price Prediction

- **Endpoint**: `/predict-housing`
- **Method**: POST
- **Payload** (example):
  ```json
  {
    "features": [
      7420,      // area
      4,         // bedrooms
      2,         // bathrooms
      3,         // stories
      2,         // parking
      1855.0,    // area_per_bed
      6,         // rooms_total
      1,         // mainroad (1=yes)
      0,         // guestroom (0=no)
      0,         // basement (0=no)
      0,         // hotwaterheating (0=no)
      1,         // airconditioning (1=yes)
      1,         // prefarea (1=yes)
      "furnished"
    ]
  }
  ```
- **Response**:
  ```json
  {
    "prediction": 8348336
  }
  ```

### Housing Assessment Questions

- **Endpoint**: `/ask-housing`
- **Method**: GET
- **Response**: List of questions to collect necessary information for prediction

## 🔧 Technologies Used

- **Flask**: API framework
- **scikit-learn**: ML models
- **Docker**: Containerization
- **pandas**: Data manipulation
- **pickle**: Model serialization

## 📝 Notes

This project addresses MLOps deployment exercises including:
- Confidence scores for predictions
- Support for multiple inputs
- Input validation
- Health check endpoint
- Docker containerization

