import streamlit as st
import pickle
import numpy as np

# Sayfa başlığı
st.set_page_config(page_title="Churn Prediction", page_icon="📊")
st.title("📞 Müşteri Kaybı (Churn) Tahmini")

# Model ve scaler yükleme
@st.cache_data
def load_model_and_scaler():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# Kullanıcıdan girişler al
st.subheader("Müşteri Bilgilerini Girin:")

tenure = st.slider("Tenure (ay)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Aylık Ücret", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Toplam Ücret", min_value=0.0, max_value=10000.0, value=1000.0)
contract = st.selectbox("Sözleşme Türü", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("İnternet Servisi", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Güvenlik", ["Yes", "No"])

# Kategorik değişkenleri sayısala çevir (örnek)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
security_map = {"No": 0, "Yes": 1}

# Özellik vektörü
features = np.array([[
    tenure,
    monthly_charges,
    total_charges,
    contract_map[contract],
    internet_map[internet_service],
    security_map[online_security]
]])

# Veriyi ölçekle
features_scaled = scaler.transform(features)

# Tahmin yap
if st.button("Tahmin Et"):
    prediction = model.predict(features_scaled)
    if prediction[0] == 1:
        st.error("⚠️ Bu müşteri churn edebilir!")
    else:
        st.success("✅ Bu müşteri büyük ihtimalle kalacak.")

