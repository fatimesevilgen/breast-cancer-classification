import streamlit as st
from predict import predict 
from sklearn.datasets import load_breast_cancer

st.title("🔬 Göğüs Kanseri Tahmin Uygulaması")
st.write("Bu uygulama, girdiğiniz özelliklere göre göğüs kanseri olup olmadığınızı tahmin eder.")

data = load_breast_cancer()
feature_names = data.feature_names

input_labels = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

st.sidebar.header("🔢 Özellik Girişi")
input_data = {}
for label in input_labels:
    mean_val = data.data[:, list(feature_names).index(label)]
    val = st.sidebar.number_input(label=label, value=float(mean_val.mean()), format="%.4f")
    input_data[label] = val


if st.button("Tahmin Yap"):
    result = predict(**input_data)
    st.subheader("📊 Tahmin Sonucu:")
    st.write(f"**{result['prediction']}**")
    st.write(f"Tahmin Olasılıkları: Malign: {result['probability']['Malign']} | Benign: {result['probability']['Benign']}")
