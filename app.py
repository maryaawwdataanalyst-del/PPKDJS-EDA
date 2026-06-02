import streamlit as st
import joblib
import numpy as np
import pandas as pd
 
# ---------- Konfigurasi halaman ----------
st.set_page_config(
    page_title='Prediksi Regresi Linear',
    page_icon=':bar_chart:',
    layout='centered',
)
 
# ---------- Muat model & scaler (cached) ----------
@st.cache_resource
def load_artefak():
    model  = joblib.load('regresi_berganda.pkl')
    scaler = joblib.load('scaler.pkl')
    fitur  = joblib.load('fitur.pkl')
    return model, scaler, fitur
 
model, scaler, FITUR = load_artefak()
 
# ---------- Header ----------
st.title(':bar_chart: Web Prediksi Regresi Linear')
st.markdown('Masukkan nilai tiap fitur di sidebar, lalu klik **Prediksi**.')
st.divider()
 
# ---------- Input di sidebar ----------
st.sidebar.header('Input Fitur')
input_user = {}
for f in FITUR:
    input_user[f] = st.sidebar.number_input(
        label=f,
        value=0.0,
        step=0.1,
        format='%.4f',
    )
 
# ---------- Tombol prediksi ----------
if st.sidebar.button('Prediksi', type='primary', use_container_width=True):
    try:
        # Susun DataFrame sesuai urutan FITUR (hindari warning feature names)
        nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
        nilai_sc = scaler.transform(nilai)
        pred = model.predict(nilai_sc)[0]
 
        # Tampilkan hasil
        st.success(f'Hasil prediksi:  **{pred:,.4f}**')
 
        # Tampilkan input yang dipakai
        st.subheader('Input yang Digunakan')
        st.dataframe(pd.DataFrame([input_user]), use_container_width=True)
 
        # Tampilkan koefisien model (untuk transparansi)
        st.subheader('Koefisien Model (Terstandarisasi)')
        df_koef = pd.DataFrame({
            'Fitur': FITUR,
            'Koefisien': model.coef_.round(4),
        })
        st.dataframe(df_koef, use_container_width=True, hide_index=True)
        st.caption(f'Intercept (β₀) = {model.intercept_:.4f}')
 
    except Exception as e:
        st.error(f'Terjadi error: {e}')
else:
    st.info('Isi nilai fitur di sidebar, lalu klik tombol Prediksi.')
 
# ---------- Footer ----------
st.divider()
st.caption('Dibuat untuk PPKD Jakarta Selatan — Kejuruan Data Analyst')
