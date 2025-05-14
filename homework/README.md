# mldeployment-cpe393

# model export
Run train.py. (model.pkl will be saved in app folder)

# Go to the directory in terminal
cd "project folder directory"

# Build Docker image
docker build -t ml-model .

# Run Docker container
docker run -p 9000:9000 ml-model

# Test the API in new terminal

curl -X POST http://localhost:9000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

expected output

{"prediction": 0}



