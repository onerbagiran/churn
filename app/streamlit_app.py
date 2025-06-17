import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Modeli ve scaler'ı yükle
try:
    with open('C:/Users/IT/Desktop/churn-project/models/xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('C:/Users/IT/Desktop/churn-project/models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f"Model veya scaler yüklenirken hata oluştu: {e}")
    st.stop()

# Streamlit uygulaması başlığı
st.title("Müşteri Churn Tahmin Uygulaması")
st.write("Müşteri bilgilerini girerek churn olasılığını tahmin edin.")

# Giriş formu oluştur
with st.form("churn_form"):
    st.subheader("Müşteri Bilgilerini Girin")

    # Sayısal özellikler
    senior_citizen = st.selectbox("Yaşlı Vatandaş (Senior Citizen)", [0, 1], help="0: Hayır, 1: Evet")
    tenure = st.slider("Hizmet Süresi (Ay)", 0, 72, 12, help="Müşterinin hizmet aldığı ay sayısı")
    monthly_charges = st.number_input("Aylık Ücret ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("Toplam Ücret ($)", min_value=0.0, max_value=10000.0, value=600.0)

    # İkili özellikler
    partner = st.selectbox("Eş/Partner", [0, 1], help="0: Hayır, 1: Evet")
    dependents = st.selectbox("Bakmakla Yükümlü Olduğu Kişi", [0, 1], help="0: Hayır, 1: Evet")
    phone_service = st.selectbox("Telefon Servisi", [0, 1], help="0: Hayır, 1: Evet")
    paperless_billing = st.selectbox("Kağıtsız Fatura", [0, 1], help="0: Hayır, 1: Evet")
    online_security = st.selectbox("Çevrimiçi Güvenlik", [0, 1], help="0: Hayır, 1: Evet")
    online_backup = st.selectbox("Çevrimiçi Yedekleme", [0, 1], help="0: Hayır, 1: Evet")
    device_protection = st.selectbox("Cihaz Koruması", [0, 1], help="0: Hayır, 1: Evet")
    tech_support = st.selectbox("Teknik Destek", [0, 1], help="0: Hayır, 1: Evet")
    streaming_tv = st.selectbox("TV Yayını", [0, 1], help="0: Hayır, 1: Evet")
    streaming_movies = st.selectbox("Film Yayını", [0, 1], help="0: Hayır, 1: Evet")

    # Kategorik özellikler
    internet_service = st.selectbox("İnternet Servisi", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Sözleşme Türü", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Ödeme Yöntemi", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    gender = st.selectbox("Cinsiyet", ["Female", "Male"])
    multiple_lines = st.selectbox("Çoklu Hat", ["No", "Yes", "No phone service"])

    # Formu gönder butonu
    submitted = st.form_submit_button("Tahmin Yap")

# Form gönderildiğinde tahmin yap
if submitted:
    # Giriş verilerini bir DataFrame'e dönüştür
    input_data = {
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        'gender_Male': 1 if gender == "Male" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0
    }

    input_df = pd.DataFrame([input_data])

    # Veriyi ölçeklendir
    input_scaled = scaler.transform(input_df)

    # Tahmin yap
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Sonuçları göster
    st.subheader("Tahmin Sonuçları")
    if prediction[0] == 1:
        st.error("Bu müşteri churn etme olasılığı yüksek! 🚨")
    else:
        st.success("Bu müşteri churn etme olasılığı düşük. ✅")

    st.write(f"Churn Etmeme Olasılığı: %{prediction_proba[0]*100:.2f}")
    st.write(f"Churn Etme Olasılığı: %{prediction_proba[1]*100:.2f}")

    # Olasılıkları görselleştir
    st.subheader("Churn Olasılık Grafiği")
    chart_data = pd.DataFrame({
        'Kategori': ['Churn Etmez', 'Churn Eder'],
        'Olasılık': [prediction_proba[0], prediction_proba[1]]
    })

    # Streamlit ile çubuk grafik oluştur
    st.bar_chart(chart_data.set_index('Kategori'))