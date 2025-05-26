import joblib

model = joblib.load('knn_breast_cancer_model.pkl')
scaler = joblib.load('scaler_breast_cancer.pkl')

def preprocess(**kwargs):
    ordered_values = [kwargs[key] for key in [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]]
    return scaler.transform([ordered_values])

def predict(**kwargs):
    X_input_scaled = preprocess(**kwargs)
    prediction = model.predict(X_input_scaled)[0]
    proba = model.predict_proba(X_input_scaled)[0]

    result = {
        'prediction': 'Benign (İyi Huylu)' if prediction == 1 else 'Malign (Kötü Huylu)',
        'probability': {
            'Malign': f"{proba[0]:.2%}",
            'Benign': f"{proba[1]:.2%}"
        }
    }
    return result
