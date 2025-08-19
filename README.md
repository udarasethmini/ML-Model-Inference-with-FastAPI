# ML-Model-Inference-with-FastAPI
ML Model Inference with FastAPI - Iris Classification API
Problem Description
This project deploys a machine learning model for Iris Flower Classification as a web API using FastAPI. The model predicts the species of an iris flower (setosa, versicolor, or virginica) based on four input features: sepal length, sepal width, petal length, and petal width. The dataset used is the built-in Iris dataset from scikit-learn.
Model Choice Justification
I chose LogisticRegression from scikit-learn for this multiclass classification problem. It's simple, interpretable, and performs well on this small, linearly separable dataset (achieving ~97-100% accuracy on the test set). More complex models like RandomForest were considered but not necessary for this straightforward task.
API Usage Examples
The API has three endpoints:

GET /: Health check.
POST /predict: Accepts JSON input with features and returns the predicted species and confidence.
GET /model-info: Returns metadata about the model.

Example Requests

Health Check (using curl):
curl http://localhost:8000/

Response: {"status": "healthy", "message": "ML Model API is running"}

Prediction (using curl):
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

Response: {"prediction": "setosa", "confidence": 0.981629253145981} (example; actual confidence may vary slightly)
Another example input (for versicolor): {"sepal_length": 6.3, "sepal_width": 2.5, "petal_length": 4.9, "petal_width": 1.5}


Test more examples via the interactive docs at http://localhost:8000/docs.
How to Run the Application

Install dependencies: pip install -r requirements.txt
Run the notebook (iris_classification_api.ipynb) to generate model.pkl and main.py.
Start the server: uvicorn main:app --reload
Access the API at http://localhost:8000. Use http://localhost:8000/docs for testing.

Assumptions and Limitations

No data preprocessing (e.g., scaling) is applied, as it's not needed for this model/dataset.
The model is basic and trained on a small dataset; it may not generalize to noisy real-world data.
Confidence is the maximum probability from the model's predict_proba.
Error handling covers invalid inputs, but very extreme values may lead to unrealistic predictions.
