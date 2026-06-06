import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title='Classifier Tabular', page_icon=':bar_chart:', layout='wide')

@st.cache_resource
def load_artefak():
    # Tambahkan pengecekan file agar tidak error jika file hilang
    files = ['lr_best.pkl', 'preprocessor.pkl', 'selector.pkl', 'label_encoder.pkl', 'meta.pkl', 'treshold.txt']
    for f in files:
        if not os.path.exists(f):
            st.error(f"File {f} tidak ditemukan!")
            st.stop()
            
    model = joblib.load('lr_best.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    selector = joblib.load('selector.pkl')
    le = joblib.load('label_encoder.pkl')
    meta = joblib.load('meta.pkl')
    with open('treshold.txt') as f:
        thr = float(f.read().strip())
    return lr_best, preprocessor, selector, le, meta, tr

# Load artefak
model, preprocessor, selector, le, meta, threshold = load_artefak()
NUM_COLS = meta['NUM_COLS']
CAT_COLS = meta['CAT_COLS']
# Asumsi meta['CAT_CATEGORIES'] berisi dict {nama_kolom: [list_kategori]}
CAT_CATEGORIES = meta.get('CAT_CATEGORIES', {}) 

st.title(':bar_chart: Web Klasifikasi Tabular')
st.caption(f'Threshold prediksi: {threshold:.3f}  |  Kelas: {list(le.classes_)}')
st.divider()

# Form input fitur
st.subheader('Masukkan nilai fitur:')
col1, col2 = st.columns(2)
input_user = {}

with col1:
    st.markdown('**Fitur Numerik**')
    for kol in NUM_COLS:
        input_user[kol] = st.number_input(
            label=kol, value=0.0, step=0.1, format='%.4f', key=f'num_{kol}'
        )

with col2:
    st.markdown('**Fitur Kategorikal**')
    for kol in CAT_COLS:
        # Perbaikan: Gunakan st.selectbox jika kategori tersedia di meta
        if kol in CAT_CATEGORIES:
            input_user[kol] = st.selectbox(label=kol, options=CAT_CATEGORIES[kol], key=f'cat_{kol}')
        else:
            input_user[kol] = st.text_input(label=kol, value='', key=f'cat_{kol}')

st.divider()

if st.button('Prediksi', type='primary', use_container_width=True):
    try:
        df_input = pd.DataFrame([input_user])
        
        # Prediksi
        X_enc = preprocessor.transform(df_input)
        X_sel = selector.transform(X_enc)
        
        proba = model.predict_proba(X_sel)[0, 1]
        pred = int(proba >= threshold)
        kelas_pred = le.classes_[pred]

        st.success(f'Hasil prediksi: **{kelas_pred}**')

        cm1, cm2 = st.columns(2)
        cm1.metric('Probabilitas kelas positif', f'{proba:.4%}')
        cm2.metric('Threshold', f'{threshold:.4f}')
        st.progress(float(proba))

        with st.expander("Lihat detail input"):
            st.dataframe(df_input, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f'Terjadi kesalahan saat memproses data: {e}')

st.divider()
st.caption('Dibuat untuk PPKD Jakarta Selatan - Kejuruan Data Analyst')
